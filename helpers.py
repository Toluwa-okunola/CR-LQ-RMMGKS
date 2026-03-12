from copy import deepcopy


import matplotlib.pyplot as plt
import numpy as np
import astra
from trips.utilities.phantoms import *
from venv import create
import pylops
from trips.solvers.MMGKS import *
from trips.utilities.helpers import *
from trips.utilities.operators import *

from typing import Optional, Dict, Any, List
from skimage.transform import resize
from datetime import datetime


from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator

import pickle
from typing import List, Tuple, Optional

def vec(u):
    """Vectorize a 2D array."""
    return u.reshape((-1, 1))


def vectorize_func(arrays):
    """Vectorize multiple arrays and stack them."""
    return np.vstack([vec(arr) for arr in arrays])


def gen_blocks(shape: Tuple[int, int], t_end: int = 3, v_max: int = 2, 
               v_min: int = 1, padding: int = 4, add: float = 0):
    """
    Generate a sequence of images with moving blocks.
    
    Args:
        shape: (height, width) of the image
        t_end: Number of time frames
        v_max: Maximum velocity
        v_min: Minimum velocity  
        padding: Padding around edges
        add: Base value to add to all pixels
        
    Returns:
        test_sequence object with trajectories and velocities
    """
    v_mag_ = v_min
    v_mag = v_max
    us = []

    v_max = padding
    v_min = padding
    
    for t in range(t_end):
        u = np.zeros(shape) + add
        scale = shape[0] // 8
        size = u.shape[0]
        
        u[v_max+v_mag*t:2*scale+v_max+v_mag*t, v_max:2*scale+v_max] = 1
        
        u[size-v_min-1*scale - v_mag_*t:size-v_min - v_mag_*t, 
          size-v_min-2*scale:size-v_min] = 0.99
        u[size-v_min-1*scale-scale - v_mag_*t:size-v_min-scale-v_mag_*t,
          size-v_min-2*scale:size-v_min-scale] = 0.99

        u[v_min + v_mag_*t:v_min+1*scale+ v_mag_*t,
          size-v_min-2*scale-v_mag_*t:size-v_min- v_mag_*t] = 0.98
        u[v_min+scale+ v_mag_*t:v_min+2*scale+ v_mag_*t,
          size-v_min-2*scale- v_mag_*t:size-v_min-scale-v_mag_*t] = 0.98

        u[size-v_max-2*scale:size-v_max-1*scale,
          v_max+ v_mag*t:v_max+3*scale+ v_mag*t] = 0.97
        u[size-v_max-1*scale:size-v_max,
          v_max+scale+ v_mag*t:v_max+2*scale+ v_mag*t] = 0.97

        us.append(u)
    
    def gen_v(u):
        v = np.zeros(shape + (2,))
        v[np.where(u==1)[0].min()-v_mag:np.where(u==1)[0].max()+v_mag+1,
          np.where(u==1)[1].min()-v_mag:np.where(u==1)[1].max()+v_mag+1] = np.array([1,0])*v_mag
        v[np.where(u==0.99)[0].min()-v_mag_:np.where(u==0.99)[0].max()+v_mag_+1,
          np.where(u==0.99)[1].min()-v_mag_:np.where(u==0.99)[1].max()+v_mag_+1] = np.array([-1,0])*v_mag_
        v[np.where(u==0.98)[0].min()-v_mag_:np.where(u==0.98)[0].max()+v_mag_+1,
          np.where(u==0.98)[1].min()-v_mag_:np.where(u==0.98)[1].max()+v_mag_+1] = np.array([1,-1])*v_mag_
        v[np.where(u==0.97)[0].min()-v_mag:np.where(u==0.97)[0].max()+v_mag+1,
          np.where(u==0.97)[1].min()-v_mag:np.where(u==0.97)[1].max()+v_mag+1] = np.array([0,1])*v_mag
        return v

    def gen_v_prime(u):
        v = np.zeros(shape + (2,))
        v[np.where(u==1)[0].min()-v_mag:np.where(u==1)[0].max()+v_mag+1,
          np.where(u==1)[1].min()-v_mag:np.where(u==1)[1].max()+v_mag+1] = -np.array([1,0])*v_mag
        v[np.where(u==0.99)[0].min()-v_mag_:np.where(u==0.99)[0].max()+v_mag_+1,
          np.where(u==0.99)[1].min()-v_mag_:np.where(u==0.99)[1].max()+v_mag_+1] = -np.array([-1,0])*v_mag_
        v[np.where(u==0.98)[0].min()-v_mag_:np.where(u==0.98)[0].max()+v_mag_+1,
          np.where(u==0.98)[1].min()-v_mag_:np.where(u==0.98)[1].max()+v_mag_+1] = -np.array([1,-1])*v_mag_
        v[np.where(u==0.97)[0].min()-v_mag:np.where(u==0.97)[0].max()+v_mag+1,
          np.where(u==0.97)[1].min()-v_mag:np.where(u==0.97)[1].max()+v_mag+1] = -np.array([0,1])*v_mag
        return v
    
    us_rev = us[::-1]
    vs = []
    v_primes = []
    for t in range(0, t_end-1):
        vs.append(gen_v(us[t]))
        v_primes.append(gen_v_prime(us_rev[t]))

    u_traj = [vec(u) for u in us] 
    u_inv_traj = [vec(u_inv) for u_inv in us_rev]
    
    class TestSequence:
        def __init__(self, u_traj, u_inv_traj, vs, v_primes):
            self.u_traj = u_traj
            self.u_inv_traj = u_inv_traj
            self.vs = vs
            self.v_primes = v_primes
        
    return TestSequence(u_traj, u_inv_traj, vs, v_primes)


def dynamic_shepp_logan(N: int, nt: int = 20, moving_ellipse_idx: int = 4, 
                       motion_type: str = "rotation", add: float = 0.1):
    """
    Generate a dynamic Shepp-Logan phantom with one moving ellipse.
    
    Args:
        N: Resolution (NxN image)
        nt: Number of time frames
        moving_ellipse_idx: Index of ellipse to animate (default 4 = eye region)
        motion_type: Type of motion ("translation", "rotation", or "combined")
        add: Base value to add to phantom
        
    Returns:
        frames: List of numpy arrays, each of size (N, N)
    """
    # Original Shepp-Logan ellipse parameters
    #                  A      a      b     x0      y0    phi
    e = np.array([
        [  1,    .69,   .92,    0,       0,   0 ], 
        [-.8,  .6624, .8740,    0,  -.0184,   0 ],
        [-.2,  .1100, .3100,  .22,       0,  -18],
        [-.2,  .1600, .4100, -.22,       0,   18],
        [ .1,  .2100, .2500,    0,     .35,   0 ],
        [ .1,  .0460, .0460,    0,      .1,   0 ],
        [ .1,  .0460, .0460,    0,     -.1,   0 ],
        [ .1,  .0460, .0230, -.08,   -.605,   0 ],
        [ .1,  .0230, .0230,    0,   -.606,   0 ],
        [ .1,  .0230, .0460,  .06,   -.605,   0 ] 
    ])
    
    # Create coordinate system
    xn = ((np.arange(0, N) - (N-1)/2) / ((N-1)/2))
    Xn = np.tile(xn, (N, 1))
    Yn = np.rot90(Xn)
    
    frames = []
    
    # Define movement parameters for the selected ellipse
    original_x0 = e[moving_ellipse_idx, 3]
    original_y0 = e[moving_ellipse_idx, 4]
    
    # Movement pattern optimized for different motion types
    if moving_ellipse_idx == 4:  # Eye region
        movement_radius = 0.08  # Smaller movement for eyes
        rotation_amplitude = (N//64) * 3  # degrees
        movement_pattern = "horizontal"  # Eyes typically move more horizontally
    else:
        movement_radius = 0.15
        rotation_amplitude = 30  # degrees
        movement_pattern = "circular"
    
    for frame_idx in range(nt):
        X = np.zeros((N, N))
        
        # Calculate movement for this frame based on pattern
        if frame_idx == 0:
            # First frame is always the original phantom
            move_x = 0
            move_y = 0
            rotation_angle = 0
        else:
            # For subsequent frames, calculate movement
            angle = 2 * np.pi * (frame_idx) / (nt - 1) if nt > 1 else 0
            
            # Calculate translational movement
            if motion_type in ["translation", "combined"]:
                if movement_pattern == "horizontal":
                    move_x = movement_radius * np.sin(angle)
                    move_y = movement_radius * 0.3 * np.cos(2 * angle)
                else:
                    move_x = movement_radius * np.cos(angle)
                    move_y = movement_radius * np.sin(angle)
            else:
                move_x = 0
                move_y = 0
            
            # Calculate rotational movement
            if motion_type in ["rotation", "combined"]:
                rotation_angle = rotation_amplitude * np.sin(angle) * np.pi / 180
            else:
                rotation_angle = 0
        
        # For each ellipse to be added     
        nn = len(e)
        for i in range(nn):
            A   = e[i, 0]
            a2  = e[i, 1]**2
            b2  = e[i, 2]**2
            
            # Apply movement to the selected ellipse
            if i == moving_ellipse_idx:
                x0  = original_x0 + move_x
                y0  = original_y0 + move_y
            else:
                x0  = e[i, 3]
                y0  = e[i, 4]
            
            # Apply rotation to the selected ellipse
            if i == moving_ellipse_idx:
                phi = (e[i, 5] + rotation_angle * 180 / np.pi) * np.pi / 180
            else:
                phi = (e[i, 5] + rotation_angle * 180 / np.pi) * np.pi / 180
            
            x   = Xn - x0
            y   = Yn - y0
            idd = ((x*np.cos(phi) + y*np.sin(phi))**2)/a2 + \
                  ((y*np.cos(phi) - x*np.sin(phi))**2)/b2
            idx = np.where(idd <= 1)

            # Add the amplitude of the ellipse
            X[idx] += A
        
        # Ensure non-negative values
        idx = np.where(X < 0)
        X[idx] = 0
        
        frames.append(X + add)
    
    return frames


def gen_pinball(nx: int, ny: int, nt: int, add: float = 0):
    """
    Generate a pinball animation sequence.
    
    Args:
        nx, ny: Image dimensions
        nt: Number of frames
        add: Base value to add to images
        
    Returns:
        frames: List of frame arrays
    """
    width, height = nx, ny
    num_frames = 40

    # Derived oval and ball parameters
    oval_center = (width // 2, height // 2)
    oval_axes = (int(0.4 * width), int(0.2 * height))
    ball_radius = width / 12

    def generate_frame(ball_x):
        img = np.zeros((height, width), dtype=np.uint8) + int(add * 256) 
        
        # Create the oval
        for y in range(height):
            for x in range(width):
                if ((x - oval_center[0]) ** 2) / oval_axes[0]**2 + \
                   ((y - oval_center[1]) ** 2) / oval_axes[1]**2 <= 1:
                    img[y, x] = int((add + 0.3) * 256)
        
        # Create the ball
        for y in range(height):
            for x in range(width):
                if (x - ball_x) ** 2 + (y - oval_center[1]) ** 2 <= ball_radius**2:
                    img[y, x] = 255
        
        return img

    # Generate frames
    frames = []
    for i in range(num_frames):
        ball_x = int(oval_center[0] - oval_axes[0] + ball_radius + 
                    (2 * (oval_axes[0] - ball_radius) * i / (num_frames - 1)))
        frame = generate_frame(ball_x)
        frames.append(frame.reshape(-1, 1) / 255)

    return frames[(num_frames - nt) // 2:(num_frames - nt) // 2 + nt]


def gen_shepp_logan(nx: int, nt: int, move_idx: int = 2, 
                   dx_total: float = 0.1, add: float = 0):
    """
    Generate a sequence of dynamic Shepp-Logan phantoms with one moving ellipse.

    Args:
        nx: Image resolution (NxN)
        nt: Number of frames
        move_idx: Index of the ellipse to move (0-based)
        dx_total: Total movement along x-direction (from start to end)
        add: Base value to add to phantom

    Returns:
        frames: List of (N,N) arrays, each a dynamic Shepp-Logan phantom
    """
    N = nx
    # Static ellipses (original Shepp-Logan)
    e_base = np.array([
        [  1,  .69,  .92,    0,       0,   0 ],
        [-.8, .6624,.8740,   0,  -.0184,   0 ],
        [-.2, .1100,.3100,  .22,     0,  -18],
        [-.2, .1600,.4100, -.22,     0,   18],
        [ .1, .2100,.2500,   0,     .35,   0 ],
        [ .1, .0460,.0460,   0,      .1,   0 ],
        [ .1, .0460,.0460,   0,     -.1,   0 ],
        [ .1, .0460,.0230, -.08,  -.605,   0 ],
        [ .1, .0230,.0230,   0,   -.606,   0 ],
        [ .1, .0230,.0460,  .06,  -.605,   0 ]
    ])

    # Normalized coordinates
    xn = ((np.arange(0, N) - (N - 1) / 2) / ((N - 1) / 2))
    Xn = np.tile(xn, (N, 1))
    Yn = np.rot90(Xn)

    frames = []

    for t in range(nt):
        # Compute shifted ellipses
        e = e_base.copy()
        shift = dx_total if nt == 1 else dx_total * (t / (nt - 1))
        e[move_idx, 3] += shift  # Modify x0 of chosen ellipse

        X = np.zeros((N, N))
        for i in range(e.shape[0]):
            A, a, b, x0, y0, phi = e[i]
            a2 = a ** 2
            b2 = b ** 2
            phi_rad = phi * np.pi / 180
            x = Xn - x0
            y = Yn - y0
            idd = ((x * np.cos(phi_rad) + y * np.sin(phi_rad)) ** 2) / a2 + \
                  ((y * np.cos(phi_rad) - x * np.sin(phi_rad)) ** 2) / b2
            mask = idd <= 1
            X[mask] += A

        X[X < 0] = 0
        frames.append(X + add)

    return frames

def tectonic(N):
    # Creates a tectonic phantom of size N x N
    x   = np.zeros((N,N))
    N5  = round(N/5)
    N13 = round(N/13)
    N7  = round(N/7)
    N20 = round(N/20)

    # The right plate
    xr = np.arange(N5,N5+N7+1)-1
    yr = np.arange(5*N13,N+1)-1
    x[np.ix_(xr, yr)] = 0.75

    # The angle of the right plate
    i = N5-1
    for j in range(N20+1):
        if ((j+1)%2) != 0:
            i -= 1
            x[i, 5*N13+j:] = 0.75

    # The left plate before the break
    xr = np.arange(N5,N5+N5+1)-1
    yr = np.arange(1,5*N13+1)-1
    x[np.ix_(xr, yr)] = 1

    # The break from the left plate
    rang = np.arange(5*N13, min(12*N13,N)+1)-1
    for j in rang:
        if ((j+1)%2) != 0:
            xr += 1
        x[xr,j] = 1
    
    return x

#=========================================================================
def smooth(N, p=4):
    # SMOOTH Creates a 2D test image of a smooth function
    xx    = np.arange(1,N+1)-1
    I, J  = np.meshgrid(xx,xx, indexing='xy')
    sigma = 0.25*N
    #
    c = np.array([[0.6*N, 0.6*N], [0.5*N, 0.3*N], [0.2*N, 0.7*N], [0.8*N, 0.2*N]])
    a = np.array([1, 0.5, 0.7, 0.9])
    x = np.zeros((N,N))
    for i in range(p):
        x += a[i]*np.exp( -(I-c[i,0])**2/(1.2*sigma)**2 - (J-c[i,1])**2/sigma**2 )
    x = x/x.max()
    
    return x

def threephases(N, p=70):
    # THREEPHASES Creates a 2D test image with three different phases
    # 1st
    xx     = np.arange(1,N+1)-1
    I, J   = np.meshgrid(xx,xx, indexing='xy')
    sigma1 = 0.025*N
    c1     = np.random.rand(p,2)*N
    x1     = np.zeros((N,N))
    for i in range(p):
        x1 += np.exp(-abs(I-c1[i,0])**3/(2.5*sigma1)**3 - abs(J-c1[i,1])**3/sigma1**3)

    t1 = 0.35
    x1[x1 < t1]  = 0
    x1[x1 >= t1] = 2

    # 2nd
    sigma2 = 0.03*N
    c2     = np.random.rand(p,2)*N
    x2     = np.zeros((N,N))
    for i in range(p):
        x2 += np.exp(-(I-c2[i,0])**2/(2*sigma2)**2 - (J-c2[i,1])**2/sigma2**2)
    
    t2 = 0.55
    x2[x2 < t2]  = 0
    x2[x2 >= t2] = 1

    # combine the two images
    x = x1 + x2
    x[x == 3] = 1
    x = x/x.max()
    
    return x

def grains(N, numGrains):
    # numGrains = int(round(3*np.sqrt(N)))
    
    # GRAINS Creates a test image of Voronoi cells
    dN        = round(N/10)
    Nbig      = N + 2*dN
    total_dim = Nbig**2

    # random pixels whose coordinates (xG,yG,zG) are the "centre" of the grains
    xG = np.ceil(Nbig*np.random.rand(numGrains,1))
    yG = np.ceil(Nbig*np.random.rand(numGrains,1))

    # set up voxel coordinates for distance computation
    xx   = np.arange(1,Nbig+1)
    X, Y = np.meshgrid(xx,xx, indexing='xy')
    X    = X.flatten(order='F')
    Y    = Y.flatten(order='F')

    # for centre pixel k [xG(k),yG(k),zG(k)] compute the distance to all the 
    # voxels in the box and store the distances in column k.
    distArray = np.zeros((total_dim,numGrains))
    for k in range(numGrains):
        distArray[:,k] = (X-xG[k])**2 + (Y-yG[k])**2

    # determine to which grain each of the voxels belong. This is found as the
    # centre with minimal distance to the given voxel
    minIdx = np.argmin(distArray, axis=1)

    # reshape to 2D, subtract 1 to have 0 as minimal value, extract the
    # middle part of the image, and scale to have 1 as maximum value
    x   = minIdx.reshape(Nbig,Nbig) - 1
    x   = x[np.ix_(dN+np.arange(1,N+1)-1, dN+np.arange(1,N+1)-1)]
    x   = x/x.max()
    
    return x



def ppower(N, relnz=0.65, p=2.6):#relnz=0.65, p=2.3
    #PPOWER Creates a 2D test image with patterns of nonzero pixels
    if N/2 == round(N/2):
        Nodd = False
    else: 
        Nodd = True
        N += 1
    #
    P = np.random.randn(N,N)
    # idx = np.random.permutation(200)
    # idx = (idx[:150])
    # P = P[idx,idx]
    #
    xx   = np.arange(1,N+1)
    I, J = np.meshgrid(xx,xx, indexing='xy')
    #
    U = ( ( (2*I-1)/N - 1)**2 + ( (2*J-1)/N - 1)**2 )**(-p/2)
    F = U*np.exp(2*np.pi*np.sqrt(-1+0j)*P)
    F = abs(np.fft.ifft2(F))
    f = -np.sort(-F.flatten(order='F'))   # 'descend'
    k = round(relnz*N**2)-1
    #
    F[F < f[k]] = 0
    x = F/f[0]
    if Nodd:
        x = F[1:-1,1:-1]
    
    return x

def blocks(nx):
    ny = nx
    shape = (nx,ny)
    t_end = 1

    x = gen_blocks(shape,t_end).u_traj[0]
    return x

def gen_phantom(phantom_name, N):

    phantom_dict = {
        "shepp_logan": shepp_logan,
        "tectonic": tectonic,
        "smooth": smooth,
        "threephases": threephases,
        "grains": grains,
        "ppower": ppower,
        "blocks": blocks
    }
    
    if phantom_name not in phantom_dict:
        raise ValueError(f"Unknown phantom name: {phantom_name}")
    
    # Special case for 'grains' and 'ppower' which require an additional parameter
    if phantom_name == "grains":
        numGrains = int(round(3 * np.sqrt(N)))
        return phantom_dict[phantom_name](N, numGrains)
    elif phantom_name == "ppower":
        return phantom_dict[phantom_name](N, relnz=0.65, p=2.6)
    
    return phantom_dict[phantom_name](N)


def gen_first_derivative_operator(n):
    D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
    L = (sparse.identity(n)-D).tocsr()
    L[-1] = 0
    Lx = L[0:, :]
    return Lx

def gen_first_derivative_operator_2D(nx, ny):
    D_x = gen_first_derivative_operator(nx)  # x-direction derivative operator (size nx)
    D_y = gen_first_derivative_operator(ny)  # y-direction derivative operator (size ny)
    
    # IDx handles differentiation in the x-direction
    IDx = sparse.kron(D_x, sparse.identity(ny))  # size (nx*ny, nx*ny)
    
    # DyI handles differentiation in the y-direction
    DyI = sparse.kron(sparse.identity(nx), D_y)  # size (nx*ny, nx*ny)
    
    # Stack the operators
    L = sparse.vstack((IDx, DyI))
    return L

def create_forward_op_ct(nx: int, ny: int, theta: np.ndarray, s: float, 
                         theta_pert: float = 0, s_pert: float = 0, 
                         ratio: float = 1.5):

    if isinstance(theta_pert, np.ndarray): 
        theta_pert = theta_pert.item()
    if isinstance(s_pert, np.ndarray): 
        s_pert = s_pert.item()
    
    num_angles = len(theta)
    theta_true = (np.array(theta) + theta_pert) % 360
    angles = np.deg2rad(theta_true)

    # Total distance perturbed
    R_sd = (s + 1) * nx 

    R_od = nx 
    R_so = s*nx+ s_pert * nx

    span_rad = 2 * np.arctan(1 / (2 * 2 - 1))
    det_count = int(np.rint(np.sqrt(2) * nx))
    detector_pixel_size = 1.5

    vol_geom = astra.create_vol_geom(nx, ny)
    proj_geom = astra.create_proj_geom(
        'fanflat',
        detector_pixel_size,
        det_count,
        angles,
        R_so,
        R_od
    )

    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    A_i = astra.OpTomo(proj_id)

    operatorf = lambda X: (A_i * X.reshape((nx, ny))).reshape(-1, 1)
    operatorb = lambda B: A_i.T * B.reshape((num_angles, det_count))

    OP = pylops.FunctionOperator(operatorf, operatorb, det_count * num_angles, nx * ny)
    return OP


def generate_problem(phantom_name: str = 'blocks', nx: int = 256, ny: Optional[int] = None, 
                    t_end: int = 10, v_max: int = 1, v_min: int = 1, padding: int = 3, 
                    root: str = 'results_nonlinear', add: float = 0) -> Dict[str, Any]:
    """
    Generate a phantom problem with specified parameters.
    
    Args:
        phantom_name: Type of phantom ('blocks', 'mnist', 'pinball', 'shepp_logan', 'emoji','ppower','tectonic','smooth','threephases','grains')
        nx: Image width
        ny: Image height (defaults to nx if None)
        t_end: Number of time frames
        v_max: Maximum velocity for blocks phantom
        v_min: Minimum velocity for blocks phantom
        padding: Padding for blocks phantom
        root: Root directory for results
        add: Base value to add to phantom
        
    Returns:
        Dictionary containing problem data and metadata
    """

    
    if ny is None:
        ny = nx
    shape = (nx, ny)
    size = nx * ny

    # Create default folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{root}/problem_{nx}_by_{ny}_{phantom_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    if phantom_name.lower() == 'tectonic':
        x_true = [tectonic(nx)]*t_end
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None
    elif phantom_name.lower() == 'smooth':
        x_true = [smooth(nx)]*t_end
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None
    elif phantom_name.lower() == 'threephases':
        x_true = [threephases(nx)]*t_end
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None
    elif phantom_name.lower() == 'grains':
        x_true = [grains(nx,nx)]*t_end
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None
    elif phantom_name.lower() == 'ppower':
        x_true = [ppower(nx,nx)]*t_end
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None
    elif phantom_name.lower() == 'blocks':
        data = gen_blocks(shape, t_end, v_max=v_max, v_min=v_min, padding=padding, add=add)
        u_traj = [u for u in data.u_traj]
        u_inv_traj = data.u_inv_traj
        v_primes = data.v_primes
        vs = data.vs

    elif phantom_name.lower() == 'mnist':
        # Note: This assumes mnist_test_seq.npy exists in the working directory
        try:
            u_traj_ = np.load('mnist_test_seq.npy')[:, 260, :, :][:t_end] / 255.0
            u_traj_ = [u.reshape(64,64) for u in u_traj_]
            u_traj_ = [resize(u, (nx,ny)) for u in u_traj_]
            u_traj = [vec(u) for u in u_traj_]
            for u in u_traj:
                u[np.isclose(u, 0, atol=add)] += add
            u_inv_traj = None
            v_primes = None
            vs = None
        except FileNotFoundError:
            raise FileNotFoundError("mnist_test_seq.npy not found. Please provide this file.")

    elif phantom_name.lower() == 'pinball':
        u_traj = gen_pinball(nx, ny, t_end, add=add)
        u_inv_traj = None
        v_primes = None
        vs = None

    elif phantom_name.lower() == 'shepp_logan':
        x_true = dynamic_shepp_logan(nx, nt=30, moving_ellipse_idx=4, add=add)[:t_end]
        u_traj = [vec(x_true[_]) for _ in range(t_end)]
        u_inv_traj = None
        v_primes = None
        vs = None

    elif phantom_name.lower() == 'emoji':
        # Note: This assumes emoji_tomo/recovered_images_dict exists
        try:
            with open('emoji_tomo/recovered_images_dict', 'rb') as f:
                recovered_images_dict = pickle.load(f)
            images = np.copy(np.array([recovered_images_dict["iso-of"].reshape(*[33, 128, 128])[i] 
                                     for i in range(0, 33, 2)]))[:t_end]
            images[images < 0] = 0
            images= [u.reshape(128,128) for u in images]
            images = [resize(u, (nx,ny)) for u in images]
            u_traj = [vec(u) / np.max(u) for u in images]
            for u in u_traj:
                u[np.isclose(u, 0, atol=0.1)] = 0
            for u in u_traj:
                u[np.isclose(u, 0, atol=add)] += add
            u_inv_traj = u_traj[::-1]
            v_primes = None
            vs = None
        except FileNotFoundError:
            raise FileNotFoundError("emoji_tomo/recovered_images_dict not found. Please provide this file.")
    else:
        raise ValueError(f"Unknown phantom_name: {phantom_name}")

    x_traj = u_traj

    description = f"""Problem Description:
- Phantom: {phantom_name}
- Image size: {shape[0]} x {shape[1]}
- Number of frames: {t_end}
- Output directory: {output_dir}
"""
    print(description)
    #save_description_to_file(description, output_dir, filename="phantom_description.txt")

    # Return the problem data and the folder path
    return {
        'u_traj': u_traj,
        'x_traj': x_traj,
        'shape': shape,
        'size': size,
        'phantom_name': phantom_name,
        'u_inv_traj': u_inv_traj,
        'v_primes': v_primes,
        'vs': vs,
        't_end': t_end,
        'output_dir': output_dir
    }

METHOD_NAMES = {
    (2, False): r'$\ell_2$-MMGKS',
    (1, False): r'$\ell_1$-MMGKS',
    (2, True):  r'CR-$\ell_2$-MMGKS',
    (1, True):  r'CR-$\ell_1$-MMGKS',
}

LINESTYLES = {
    r'$\ell_2$-MMGKS': ':',
    r'$\ell_1$-MMGKS': '--',
    r'CR-$\ell_2$-MMGKS': '-.',
    r'CR-$\ell_1$-MMGKS': '-',
}

MARKERS = {
    r'$\ell_2$-MMGKS': 'd',
    r'$\ell_1$-MMGKS': 's',
    r'CR-$\ell_2$-MMGKS': '^',
    r'CR-$\ell_1$-MMGKS': 'o',
}

COLORS = {
    r'CR-$\ell_1$-MMGKS': '#0072B2',  # Blue
    r'CR-$\ell_2$-MMGKS': '#D55E00',  # Vermillion
    r'$\ell_1$-MMGKS':    '#009E73',  # Bluish green
    r'$\ell_2$-MMGKS':    '#CC79A7',  # Reddish purple
}

def plot_snapshot(all_results, x_true, b, 
                  save_path="figure_snapshot.pdf"):

    nx,ny = x_true.shape



    method_order = [ METHOD_NAMES[key] for key in METHOD_NAMES.keys()] #['L2', 'L1', 'CR-L2', 'CR-L1']
    is_2d = (x_true.ndim == 2 and min(x_true.shape) > 1)

    # -----------------------------
    # Find CR-L1 snapshot
    # -----------------------------
    l1_results = all_results[(1, False)]
    l1_total_inner = l1_results['total_inner_iterations']

    snapshots = {}
    for key, results in all_results.items():
        name = METHOD_NAMES[key]
        all_x = results['history']['all_x']
        idx = min(l1_total_inner, len(all_x) - 1)
        snapshots[name] = all_x[idx]

    # -----------------------------
    # Layout
    # -----------------------------
    fig, axes = plt.subplots(
        1, 6,
        figsize=(14, 2.4),
        gridspec_kw={'hspace': 0.12, 'wspace': 0.05}
    )

    axes = np.array([axes])

    titles = [r"Ground Truth (x$_{\mathrm{true}}$)", "$b$"][::-1] + method_order
    signals = [x_true, b][::-1] + [snapshots[m].reshape(nx,ny) for m in method_order]

    # =========================================================
    # 1D CASE — Precompute limits
    # =========================================================
    if not is_2d:
        # Row 1 limits (signals)
        all_flat = [s.flatten() for s in signals]
        y_min = min(s.min() for s in all_flat)
        y_max = max(s.max() for s in all_flat)
        pad = 0.05 * (y_max - y_min + 1e-12)
        y_limits_row1 = (y_min - pad, y_max + pad)

        # Row 2 limits (differences)
        diffs_tmp = [np.zeros_like(x_true),] + \
                    [snapshots[m] - x_true for m in method_order]

        d_min = min(d.min() for d in diffs_tmp)
        d_max = max(d.max() for d in diffs_tmp)
        max_abs = max(abs(d_min), abs(d_max))
        pad = 0.05 * (max_abs + 1e-12)
        y_limits_row2 = (-max_abs - pad, max_abs + pad)

    # =========================================================
    # ROW 1 — Signals
    # =========================================================
    for j, (title, sig) in enumerate(zip(titles, signals)):
        ax = axes[0, j]
        ax.set_xticks([])
        ax.set_yticks([])

        if is_2d:
            if title == "b":
                im_signal = ax.imshow(sig, cmap='gray', aspect='auto')
            else:
                im_signal = ax.imshow(sig, cmap='gray', aspect='equal')

            ax.set_box_aspect(1)

            #ax.set_tick_
        else:
            ax.plot(sig.flatten(),
                    color='k' if title == "Ground Truth" else COLORS.get(title, 'gray'),
                    linewidth=2.0 if title != "Ground Truth" else 2.5)
            ax.set_ylim(y_limits_row1)

        ax.set_title(title,
                     fontweight='bold' if title == 'CR-L1' else None)

        # Remove clutter
        if j != 0:
            ax.set_yticklabels([])
        ax.set_xticklabels([])



    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    return fig

def plot_convergence(all_results, save_path='figure1_convergence.pdf', n = -1):
    """
    Main convergence plot showing RRE vs cumulative inner iterations.
    Shows full convergence behavior for all methods.
    """
    fig, axes = plt.subplots(1, 1, figsize=(12,5))
    
    # Find CR-L1 convergence point (in cumulative inner iterations)
    l1_results = all_results[(1, False)]
    l1_conv_iter = l1_results['total_outer_iterations']
    l1_total_inner = l1_results['total_inner_iterations']
    
    # Panel (a): RRE vs Cumulative Inner Iterations
    ax1 = axes
    
    for key, results in all_results.items():
        name = METHOD_NAMES[key]
        all_rre = results['history']['all_rre'][:n]
        cumulative_iters = np.arange(len(all_rre))
        
        lw = 2 if name == 'CR-L1' else 1.5
        alpha = 0.95 if name == 'CR-L1' else 0.8
        
        ax1.plot(cumulative_iters, all_rre,
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                linewidth=lw,
                label=name, # + (' (Proposed)' if name == 'CR-L1' else ''),
                alpha=alpha,
                zorder=10 if name == 'CR-L1' else 5)
        
    
    ax1.set_xlabel('Total Iterations', fontsize=12)
    ax1.set_ylabel('Relative Reconstruction Error', fontsize=12)
    #ax1.set_title('(a) Convergence Behavior', fontweight='bold', fontsize=14)
    #ax1.legend(loc='center right', framealpha=0.95, fontsize=12)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., fontsize=12)
    #ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
