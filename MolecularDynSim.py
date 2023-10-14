"""
Ben Ghertner
Feb 25, 2022
Water Molecules in a Box

This is a simulation of water molecules bouncing around
in a box under a Lennard Jones Potential,
Including rotations and torques based on electric dipole moments
of the water molecules.
"""

# # # # # # #    IMPORTS   # # # # # # #
import numpy as np
import vpython as vp

# # # # # # # # # # # # # # # # # # # # # 

# # # # # # #   CONSTANTS   # # # # # # #
m = 30.103 #mass of molecules - kgx10^-27
b = 2.725 #(sigma) size of repulsive circle of influence for a molecule in Angstrom
C = 4.9115e2 #Constant (epsilon) Jx10^-23
k = 1.38 #Boltzman's constant (J/K)x10^-23
g = 9.8e-14 #acceleration due to gravity Angstrom/ps^2
p = 6.2e-20 #dipole moment in C Angstrom
epsilon0 = 8.85e-45 # epsilon naught in 10^-23 N Angstrom^2/C^2
I_prime_0 = np.array(
    [[1.09, 0., 0.],
    [0., 1.95, 0.],
    [0., 0., 3.0]]
) # The inertial tensor in diagonized form in kg x Angstrom^2 x 10^-27

#The posistion to draw the oxygen and hydrogen atoms
#These posistions are only for visual purposes and are not
#the actual location of the atoms, that information is in
#the dipole moment and moment of inertia
ox_rel_pos = np.array([0, b/6, 0])
h1_rel_pos = np.array([b/6, -b/6, 0])
h2_rel_pos = np.array([-b/6, -b/6, 0])

# # # # # #   Initial Values   # # # # # #
init_n = 25 #Number of molecules
init_h = 50 #Height of Box in Angstrom
init_w = 50 #Width of Box in Angstrom
init_d = 50 #Depth of Box in Angstrom
init_dt = .01 #Length of 1 time step in ps
init_I = 100

# # # # # # # # # # # # # # # # # # # # # 

# # # # # # #   FUNCTIONS   # # # # # # #

def update(gas, orientations, omega, h, w, d, dt, I_prime):
    """
    update: This function updates the array of gas molecules
        for the next time step dt time later
    Parameters:
    gas - a nx6 numpy array holding the position and velocity
        information of the n gas molecules
    orientations - a nx3x3 numpy array holding n, 3x3 arrays representing
        the orientation of each of the molecules.
    omega - a nx3 numpy array holding the direction and speed of rotation
        for each molecule.
    h - Height of the box
    w - Width of the box
    d - Depth of the box
    dt - length of 1 time step
    I_prime - the moment of Inertia tensor for a water molecule
        in its diagonalized form

    Returns:
    gas - updated nx4 numpy array holding the position and velocity
        information of the n gas molecules.
    orientations - updated nx3x3 numpy array holding n, 3x3 arrays representing
        the orientation of each of the molecules.
    omega - updated nx3 numpy array holding the direction and speed of rotation
        for each molecule.
    """
    prev = gas.copy()
    #A fast speed limit must be imposed to stop the occasional
    #molecule moving to fast and causing the simulation to become unstable
    vmax = 0.1*b/dt
    wmax = 0.5/dt

    #Get forces between molecules
    f, Tau = forces(gas, orientations)
    #Update Velocities
    gas[:,3:] = prev[:,3:] + dt*f/m #### EQ.6
    #Enforce speed limit
    gas[:,3:] = np.where(gas[:,3:] > vmax, vmax, gas[:,3:])

    #Update posistions
    gas[:,:3] = prev[:,:3] + gas[:,3:]*dt
    
    #Check for collisions with the walls of the box
    #Left Wall min x bound
    gas[:,3] = np.where(gas[:,0] > 0., gas[:,3], -prev[:,3])
    gas[:,0] = np.where(gas[:,0] > 0., gas[:,0], prev[:,0] + gas[:,3]*dt)
    #Right Wall max x bound
    gas[:,3] = np.where(gas[:,0] < w, gas[:,3], -prev[:,3])
    gas[:,0] = np.where(gas[:,0] < w, gas[:,0], prev[:,0] + gas[:,3]*dt)
    #Bottom min y bound
    gas[:,4] = np.where(gas[:,1] > 0., gas[:,4], -prev[:,4])
    gas[:,1] = np.where(gas[:,1] > 0., gas[:,1], prev[:,1] + gas[:,4]*dt)
    #Top max y bound
    gas[:,4] = np.where(gas[:,1] < h, gas[:,4], -prev[:,4])
    gas[:,1] = np.where(gas[:,1] < h, gas[:,1], prev[:,1] + gas[:,4]*dt)    
    #min z bound
    gas[:,5] = np.where(gas[:,2] > 0, gas[:,5], -prev[:,5])
    gas[:,2] = np.where(gas[:,2] > 0, gas[:,2], prev[:,2] + gas[:,5]*dt) 
    #max z bound
    gas[:,5] = np.where(gas[:,2] < d, gas[:,5], -prev[:,5])
    gas[:,2] = np.where(gas[:,2] < d, gas[:,2], prev[:,2] + gas[:,5]*dt) 

    #Rotation and Torques
    prev_or = orientations.copy()

    #Create the transform matrix from coordinates with the z axis as the
    #axis of rotation to normal cartesian coords
    w_mag = np.sqrt(np.einsum('nk, nk -> n', omega, omega))
    #Noramlize the omega vector
    w_unit = omega/(w_mag[:,np.newaxis])
    #Cook up a perpendicular normalized vector
    w_per1 = np.ones_like(w_unit)
    w_per1[:,2] = (-w_unit[:,0] - w_unit[:,1])/w_unit[:,2]
    w_per1 = w_per1 / (np.sqrt(np.einsum('nk, nk -> n', w_per1, w_per1))[:,np.newaxis])
    #Cook up a second perpendicular normalized vector
    w_per2 = np.cross(w_unit, w_per1)
    #Create the transformation matrices from omega -> e
    A = np.stack((w_per1, w_per2, w_unit), axis=-1) #### EQ.9
    #Get the angles rotated in a time step
    thetas = w_mag*dt
    #Preform the transformation
    ors = np.einsum('nab, nbi, nik, nkj -> naj', 
                    A, rotation(thetas), np.linalg.inv(A), orientations) #### EQ.10
    #re-normalize to keep from compounding rounding errors
    ors = orthonormalize_vec(ors)
    #Calc angular acceleration and update omega
    I = np.einsum('nab, bi, nij -> naj', 
                  prev_or, I_prime, np.linalg.inv(prev_or)) #### EQ.14
    alpha = Tau - np.cross(omega, np.einsum('nij, nj -> ni', I, omega)) #### EQ.15
    alpha = np.einsum('nij, nj -> ni', np.linalg.inv(I), alpha) #### EQ.15

    w = omega + alpha*dt #### EQ.16
    w_mag = np.sqrt(np.einsum('nk, nk -> n', w, w))
    #Enforce a rotational speed limit so the simulation remains stable
    w = np.where(w_mag[:,np.newaxis] < wmax, w, w*wmax/w_mag[:,np.newaxis])

    """
    #Uncomment this block of code to check the equivalence of the above
    #computation done efficently with numpy arrays with the easier to read
    #for loop.

    for i in range(omega.shape[0]):
        #Create the transform matrix from coordinates with z axis the
        #axis of rotation to normal cartesian coords
        omegai = omega[i,:]
        omegai_mag = np.sqrt(np.dot(omegai, omegai))
        omega_unit = omegai/omegai_mag
        omega_per1 = np.array((1, 1, (-omega_unit[0] -omega_unit[1])/omega_unit[2]))
        omega_per1 = omega_per1/np.sqrt(np.dot(omega_per1, omega_per1))
        omega_per2 = np.cross(omega_unit, omega_per1)
        #Stack to make transform matrix
        Ai = np.stack((omega_per1, omega_per2, omega_unit), axis=1)
        #Get the angle rotated in one time step
        theta = omegai_mag*dt
        #Preform tranformations
        orientations[i,:,:] = np.matmul(Ai,
                              np.matmul(rotation(np.array([theta])), 
                              np.matmul(np.linalg.inv(Ai), orientations[i,:,:])))
        #Re-normalize vectors to keep from compounding rounding errors
        orientations[i,:,:] = orthonormalize(orientations[i,:,:])
        #Now Calculate the angular accel and update omega
        I_i = np.matmul(np.matmul(prev_or[i,:,:], I_prime), 
                      np.linalg.inv(prev_or[i,:,:]))
        #scale torque or molecules spin out of control
        alpha_i = np.matmul(np.linalg.inv(I_i),
                          (Tau[i,:]*torque_scale - np.cross(omegai,
                          np.matmul(I_i, omegai))))
        omega[i,:] = omegai + alpha_i * dt
    
    #Check the difference is approx 0
    print(w - omega)
    """    

    return gas, ors, w

def forces(gas, orientations):
    """
    force: This method does the bulk of the heavy calculations.
        It finds the distance from every molecule to every other molecule.
        Then calculates the force from a Leonard-Jones potential between
        any two molecules. They attract when they are farther apart than
        b with a force proportional to 1/r^-7 and repel when they are
        closer together than b with a force proportional to 1/r^-13.
        This method also calculates the torques caused by the dipole
        moments of the molecules.

    Parameters:
    gas - a nx6 numpy array holding the position and velocity
        information of the n gas molecules.
    orientations - a nx3x3 numpy array holding n, 3x3 arrays representing
        the orientation of each of the molecules.

    Returns:
    f - a nx3 numpy array with the net force in the x, y and z direction
        on each molecule
    Tau - a nx3 numpy array with the net torques on each molecule
    """
    n = gas.shape[0]
    gas_pos = gas[:,:3].copy()

    #Figure out a nxnx3 array which distance from each molecule to every other molecule
    gas_dist =  gas_pos[np.newaxis,:,:] - gas_pos[:,np.newaxis,:]
    r = np.sqrt(np.einsum('nmi, nmi -> nm', gas_dist, gas_dist))

    #Find direction
    r_hat = -np.where(r[:,:,np.newaxis] != 0., gas_dist/r[:,:,np.newaxis], 0.)
    #Now calculate the forces
    ratio = np.where(r != 0., (b/r)**6, 0.)
    f_mag = np.where(r != 0., 24*C*(2*ratio**2 - ratio)/r, 0.)

    #Calculate net forces
    f = np.sum(r_hat*f_mag[:,:,np.newaxis], 1) - np.array([0, -g*m, 0]) #### EQ.5

    P = -orientations[:,:,1]*p #In the negative y' direction

    # # # # EQ. 11 # # # #
    E_2 = (3*np.einsum('jk, jik -> ji', P, r_hat)[:,:,np.newaxis]*r_hat - P[:,np.newaxis,:])
    E_2 = np.where(r[:,:,np.newaxis] != 0., E_2/(4*np.pi*epsilon0*(r[:,:,np.newaxis])**3), 0.)
    E_2 = np.sum(E_2, axis=0)
    # # # # # # # # # # # #
    Tau_2 = np.cross(P, E_2) #### EQ.12

    """
    #Uncomment this block of code to check to equivalence of the above
    #computation done efficently with numpy arrays with the easier to read
    #set of nested for loops.

    Tau = np.zeros((n, 3))
    E = np.zeros((n, n, 3))
    for i in range(n):
        #Now the torques first calculate the electric field at any
        #molecule based on the dipole moments of all the other molecules
        for j in range(n):
            if i != j:
                p_j = -orientations[j,:,1]*p
                r_hat_ji = r_hat[j,i,:]
                r_sep = r[j,i]
                E[i,j,:] = (3*np.dot(p_j,r_hat_ji)*r_hat_ji - p_j)/(4*np.pi*epsilon0*r_sep**3)
        E_i = np.sum(E[i,:,:], axis=0)
        #Now use the dipole moment of molecule i and the total E field
        #from the other molecules to calculate the net torque
        p_i = -orientations[i,:,1]*p
        Tau[i,:] = np.cross(p_i, E_i)
    print(Tau - Tau_2)
    """

    return f, Tau_2

def orthonormalize(A):
    """
    orthonormalize: Preforms the Gram-Schmitt orthonormalization algorithm
        on a set of 3 vectors. The direction of the third basis vector
        will be left unchanged.

    Parameters:
    A - 3x3 array where each column cooresponds to one of the basis vectors

    Returns:
    A' - 3x3 array where each column cooresponds to one of the basis vector
        and the basis vectors are now orthonormal
    """
    x_pr = A[:,0]
    y_pr = A[:,1]
    z_pr = A[:,2]

    z_pr = z_pr/np.sqrt(np.dot(z_pr,z_pr))
    y_pr = y_pr - np.dot(y_pr,z_pr)/np.dot(z_pr, z_pr) * z_pr
    y_pr = y_pr/np.sqrt(np.dot(y_pr, y_pr))
    x_pr = x_pr - np.dot(x_pr, y_pr)/np.dot(y_pr, y_pr) * y_pr - np.dot(x_pr, z_pr)/np.dot(z_pr, z_pr) * z_pr    
    x_pr = x_pr/np.sqrt(np.dot(x_pr, x_pr))

    return np.stack((x_pr, y_pr, z_pr), axis=1)

def orthonormalize_vec(A):
    """
    orthonormalize_vec: Preforms the Gram-Schmitt ortho-normalization algorithm
        on a set of 3 vectors. This version is vectorized to preform the algorithm
        on n sets of vectors at the same time utilizing numpy array operations.

    Parameters:
    A - nx3x3 array where each column cooresponds to one of the basis vectors

    Returns:
    A' - nx3x3 array where each column cooresponds to one of the basis vector
        and the basis vectors are now orthonormal
    """
    x_pr = A[:,:,0]
    y_pr = A[:,:,1]
    z_pr = A[:,:,2]
    
    z_pr = z_pr/(np.sqrt(np.einsum('nk, nk -> n',z_pr,z_pr))[:,np.newaxis])
    y_pr = y_pr - (np.einsum('nk, nk -> n',y_pr,z_pr)[:,np.newaxis]) * z_pr
    y_pr = y_pr/(np.sqrt(np.einsum('nk, nk -> n',y_pr, y_pr))[:,np.newaxis])
    x_pr = x_pr - (np.einsum('nk, nk -> n', x_pr, y_pr)[:,np.newaxis]) * y_pr \
         - (np.einsum('nk, nk -> n',x_pr, z_pr)[:,np.newaxis]) * z_pr    
    x_pr = x_pr/(np.sqrt(np.einsum('nk, nk -> n',x_pr, x_pr))[:,np.newaxis])

    return np.stack((x_pr, y_pr, z_pr), axis=-1)

def rotation(theta): #### EQ. 8
    """
    rotation: This matrix rotates the around the third axis by theta degrees

    Parameters:
    theta - array of n angles of rotation

    Return:
    R - nx3x3 numpy array. n transformation matrices which rotate by theta radians
    """
    R = np.zeros((theta.shape[0], 3, 3))
    R[:,0,0] = np.cos(theta)
    R[:,0,1] = -np.sin(theta)
    R[:,1,0] = np.sin(theta)
    R[:,1,1] = np.cos(theta)
    R[:,2,2] = 1
    return R
    #return np.array([[np.cos(theta), -np.sin(theta), 0],
    #                 [np.sin(theta),  np.cos(theta), 0],
    #                 [0,              0,             1]])

# # # # # # # # # # # # # # # # # # # # #

# # # # # # #   Molecule    # # # # # # #

class h2o:
    """
    h20: Class to use to draw the h2o molecules
    """

    def __init__(self, b, center, orientation):
        """
        Parameters:
        b - mulecule radius
        center - vector giving the coordinates of the center of the molecule
        orientaiton - 3x3 array with a set of orthonormal vectors which
            represent the orientation of the molecule. The 2nd vector
            points towards the oxygen atom and the oxygen and hydrogen
            atoms are in the x-y plane
        """
        self.b = b
        self.update(center, orientation)

        
    def update_object(self):
        """
        update_object: redraw the posistion of all the atoms
        """
        ox_pos = np.matmul(self.orientation, ox_rel_pos) + self.center
        h1_pos = np.matmul(self.orientation, h1_rel_pos) + self.center
        h2_pos = np.matmul(self.orientation, h2_rel_pos) + self.center

        try:
            self.shell.pos = vp.vector(self.center[0], self.center[1], self.center[2])
            self.ox.pos = vp.vector(ox_pos[0], ox_pos[1], ox_pos[2])
            self.h1.pos = vp.vector(h1_pos[0], h1_pos[1], h1_pos[2])
            self.h2.pos = vp.vector(h2_pos[0], h2_pos[1], h2_pos[2])
        except:
            self.shell = vp.sphere(pos=vp.vector(self.center[0], self.center[1], self.center[2]),
                          radius=self.b/2, color=vp.color.blue, opacity=0.2)
            self.ox = vp.sphere(pos=vp.vector(ox_pos[0], ox_pos[1], ox_pos[2]),
                                    radius = self.b/4, color=vp.color.red)
            self.h1 = vp.sphere(pos=vp.vector(h1_pos[0], h1_pos[1], h1_pos[2]),
                                radius = self.b/6, color=vp.color.white)
            self.h2 = vp.sphere(pos=vp.vector(h2_pos[0], h2_pos[1], h2_pos[2]),
                                radius = self.b/6, color=vp.color.white)

    def update(self, center, orientation):
        """
        update: update the position and orientation of the molecule
            then call the method to redraw the molecule

        parameters:
        center - vector representing the position of the center of the
            molecule
        orientation - 3x3 array of basis vectors representing the orientaion
            of the molecule where the 2nd vector points towards the oxygen
            atom and the oxygen and hydrogen atoms are in the x-y plane
        """
        self.center = center
        self.orientation = orientation

        self.update_object()

    def delete(self):
        """
        delete: this needs to be called to remove the molecule from the simulation
        """
        self.shell.visible = False
        self.ox.visible = False
        self.h1.visible = False
        self.h2.visible = False
        del(self)

# # # # # # #   Animation   # # # # # # #

class interactive_animation:

    def __init__(self, h, w, d, dt, T=293, n=5, I_factor=1000):
        """
        Parameters:
        h - Height of the box
        w - Width of the box
        d - Depth of the box
        dt - length of 1 time step
        T - Temperature in Kelvin
        n - number of molecules
        I_factor - factor by which to scale the moment of inertia
        """
        self.h = self.h_next = h
        self.w = self.w_next = w
        self.d = self.d_next = d
        self.dt = self.dt_next = dt
        self.T = T
        self.n = self.n_next = n
        self.time = 0.
        self.step = 10
        self.I_factor = I_factor
        self.molecules = []
        self.gas, self.orientations, self.omega = self.reset_gas()
        
        self.setup_frame()
        self.start_ani()

    def reset_gas(self):        
        """
        reset_gas: Create an a nx6 array to hold the information
            about n gas molecules. Information includes cooridnates,
            (x, y, z) and velocity components, (vx, vy, vz). Also
            create nx3 array representing the angular velocity and
            a nx3x3 array representing the orientation of each molecule.
        
        Returns:
        gas - updated nx4 numpy array holding the position and velocity
            information of the n gas molecules.
        orientations - updated nx3x3 numpy array holding n, 3x3 arrays representing
            the orientation of each of the molecules.
        omega - updated nx3 numpy array holding the direction and speed of rotation
            for each molecule.
        """
        #Set up gas array - n rows with columns: 0   1   2   3   4   5
        #                                        x   y   z   vx  vy  vz
        gas = np.zeros((self.n,6))

        #initial x, y and z posistions are sampled from a uniform random dist
        gas[:,:3] = np.random.random(size=(self.n,3)) * np.array([self.w, self.h, self.d])

        #initial speeds are sampled from a chisquare distribution
        #This is the maxwell velocity distribution
        speeds = np.random.chisquare(3, size=self.n) * np.sqrt(k*self.T/m)
        #Direction is just random with uniform probability to go in any direction
        phi = np.random.random(size=self.n) * 2 * np.pi
        theta = np.random.random(size=self.n) * np.pi
        gas[:,3] = speeds * np.cos(phi) * np.sin(theta)
        gas[:,4] = speeds * np.sin(phi) * np.sin(theta)
        gas[:,5] = speeds * np.cos(theta)

        # # # # Creating the orientation matrices from EQ. 7 # # # #
        orientations = np.random.random(size=(gas.shape[0], 3, 3)) - 0.5
        orientations = orthonormalize_vec(orientations)

        #Start with very little random angular velocity
        omega = np.random.normal(size=(gas.shape[0], 3), scale=0.0001)

        for molecule in self.molecules:
            molecule.delete()
        self.molecules = []
        for i in range(gas.shape[0]):
            self.molecules.append(h2o(b, gas[i,:3], orientations[i,:,:]))
        return gas, orientations, omega

    def reset_walls(self):
        """reset_walls: If the walls have not been initialized then create them
                        If the walls already have been initialized then change
                        their posistion and size to the new values of h, w, d"""
        try:
            self.wallR.pos = vp.vector(self.w+b, self.h/2, self.d/2)
            self.wallR.size = vp.vector(0.01*self.h, self.h+b*2, self.d+b*2)
            self.wallL.pos = vp.vector(-b,self.h/2,self.d/2)
            self.wallL.size = vp.vector(0.01*self.h,self.h+b*2,self.d+b*2)
            self.wallB.pos = vp.vector(self.w/2,-b,self.d/2)
            self.wallB.size = vp.vector(self.w+b*2,0.01*self.w,self.d+b*2)
            self.wallT.pos = vp.vector(self.w/2,self.h+b,self.d/2)
            self.wallT.size = vp.vector(self.w+b*2,0.01*self.w,self.d+b*2)
            self.wallBK.pos = vp.vector(self.w/2,self.h/2,-b)
            self.wallBK.size = vp.vector(self.w+b*2,self.h+b*2,0.01*self.d)
        except:
            self.wallR = vp.box(pos=vp.vector(self.w+b, self.h/2, self.d/2),
                                size=vp.vector(0.01*self.h, self.h+b*2, self.d+b*2),
                                color=vp.color.gray(0.4))
            self.wallL = vp.box(pos=vp.vector(-b,self.h/2,self.d/2),
                                size=vp.vector(0.01*self.h,self.h+b*2,self.d+b*2),
                                color=vp.color.gray(0.4))
            self.wallB = vp.box(pos=vp.vector(self.w/2,-b,self.d/2),
                                size=vp.vector(self.w+b*2,0.01*self.w,self.d+b*2),
                                color=vp.color.gray(0.9))
            self.wallT = vp.box(pos=vp.vector(self.w/2,self.h+b,self.d/2),
                                size=vp.vector(self.w+b*2,0.01*self.w,self.d+b*2),
                                color=vp.color.gray(0.9))
            self.wallBK =vp.box(pos=vp.vector(self.w/2,self.h/2,-b),
                                size=vp.vector(self.w+b*2,self.h+b*2,0.01*self.d),
                                color=vp.color.gray(0.3))     

    def setup_frame(self):
        """setup_frame: Set up the initial simulation and 
        create user control widgets"""
        vp.scene.title = "Water Molecules in a Box                      "
        self.time_text = vp.wtext(pos=vp.scene.title_anchor, 
                                  text=f'Time: {self.time/1000:.3f} ns')
        #The walls
        self.reset_walls()

        vp.scene.camera.follow(self.wallBK)
        
        vp.scene.append_to_title("\n\n")
        #User Controls
        vp.scene.append_to_caption('\n')
        self.pause_b =  vp.button(
            pos=vp.scene.caption_anchor,
            text='Start',
            bind=self.pause
        )

        self.reset_b = vp.button(
            #pos=vp.scene.caption_anchor,
            text='Reset',
            bind=self.reset
        )
        vp.scene.append_to_caption("    Number: ")
        self.n_box = vp.winput(bind=self.update_n, text=str(self.n))
        vp.scene.append_to_caption("    Time Step: ")
        self.dt_box = vp.winput(bind=self.update_dt, text=str(self.dt))
        vp.scene.append_to_caption(" ps")

        vp.scene.append_to_caption('\n\n')
        vp.scene.append_to_caption('     Speed: ')
        self.speed_text = vp.wtext(text='x{step}'.format(step=self.step))
        vp.scene.append_to_caption('\n\n')
        vp.scene.append_to_caption('x1')
        self.speed_slider = vp.slider(
            bind=self.speed_update,
            min=1,
            max=100,
            step=1,
            value=self.step
        )
        vp.scene.append_to_caption('x100')
        vp.scene.append_to_caption('\n\n')
        vp.scene.append_to_caption('     Temperature: ')
        self.temp_text = vp.wtext(text='{temp}K'.format(temp=self.T))
        vp.scene.append_to_caption('\n\n')
        vp.scene.append_to_caption('1K')
        self.temp_slider = vp.slider(
            bind=self.temp_update,
            min=0,
            max=3,
            step=0.01,
            value=np.log10(self.T)

        )
        vp.scene.append_to_caption('1,000K\n\n\n')
        vp.scene.append_to_caption('     Moment of Inertia: ')
        self.I_text = vp.wtext(text='x{I}'.format(I=self.I_factor))
        vp.scene.append_to_caption('\n\n')
        vp.scene.append_to_caption('x1')
        self.I_slider = vp.slider(
            bind=self.I_update,
            min=0,
            max=3,
            step=0.01,
            value=np.log10(self.I_factor)

        )
        vp.scene.append_to_caption('x1,000\n\n\n')
        vp.scene.append_to_caption('Height: ')
        self.h_box = vp.winput(bind=self.update_h, text=self.h)
        vp.scene.append_to_caption(' A     Width: ')
        self.w_box = vp.winput(bind=self.update_w, text=self.w)
        vp.scene.append_to_caption(' A      Depth: ')
        self.d_box = vp.winput(bind=self.update_d, text=self.d)
        vp.scene.append_to_caption(' A')

    def start_ani(self):
        """start_ani: Starts the animation but paused"""
        self.paused = True
        #Run the animation
        while True:
            vp.rate(30)
            self.animate()

    def pause(self, b):
        """pause: Pause/unpause the simulation and change the button from Start/Pause"""
        if self.paused:
            self.pause_b.text = 'Pause'
            self.reset_b.disabled = True
        else:
            self.pause_b.text = 'Start'
            self.reset_b.disabled = False
        self.paused = not self.paused
        

    def reset(self, event):
        """reset: Get new gas array and update frame"""
        #update variables that have been changed
        self.n = self.n_next
        self.h = self.h_next
        self.w = self.w_next
        self.d = self.d_next
        self.dt = self.dt_next
        #Reset the simulation with new values
        self.gas, self.orientations, self.omega = self.reset_gas()
        self.reset_walls()
        self.time = 0.
        self.time_text.text = f'Time: {self.time/1000:.3f} ns'

    def temp_update(self, slider):
        """temp_update: Update temp to new slider value"""
        self.T = int(10**slider.value)
        self.temp_text.text = '{temp}K'.format(temp=self.T)

    def I_update(self, slider):
        """I_update: Update the factor I is increased by to new slider value"""
        self.I_factor = int(10**slider.value)
        self.I_text.text = 'x{I}'.format(I=self.I_factor)

    def speed_update(self, slider):
        """speed_update: Update the number of time steps
            computed for each frame to new slider value"""
        self.step = int(slider.value)
        self.speed_text.text = 'x{step}'.format(step=self.step)

    def update_n(self, box):
        """update_n: update the number of molecules"""
        try:
            self.n_next = int(box.text)
        except:
            print('Error, Input for \'n\' can not be interpreted as an integer.')
            box.text = str(self.n_next)

    def update_h(self, box):
        """update_h: update the height of the box"""
        try:
            self.h_next = int(box.text)
        except:
            print('Error, Input for \'height\' can not be interpreted as an integer.')
            box.text = (str(self.h_next))

    def update_w(self, box):
        """update_w: update the width of the box"""
        try:
            self.w_next = int(box.text)
        except:
            print('Error, Input for \'width\' can not be interpreted as an integer.')
            box.text = str(self.w_next)
   
    def update_d(self, box):
        """update_d: update the depth of the box"""
        try:
            self.d_next = int(box.text)
        except:
            print('Error, Input for \'depth\' can not be interpreted as an integer.')
            box.text = str(self.d_next)
    
    def update_dt(self, box):
        """update_dt: update the time step"""
        try:
            self.dt_next = float(box.text)
        except:
            print('Error, Input for \'time step\' can not be interpreted as a number.')
            box.text = (str(self.dt_next))

    def animate(self):
        """animate: Update the gas then updates the frame."""
        if not self.paused: #Do nothing if paused
            for j in range(self.step):
                with np.errstate(divide='ignore', invalid='ignore'): 
                    #suppress division by zero warnings
                    self.gas, self.orientations, self.omega = update(self.gas, 
                                    self.orientations, self.omega, self.h, self.w, 
                                    self.d, self.dt, I_prime_0*self.I_factor)
                self.time += self.dt
            self.time_text.text = f'Time: {self.time/1000:.3f} ns'
            #Update the posistion of the molecules
            for i, sphere in enumerate(self.molecules):
                sphere.update(self.gas[i,:3], self.orientations[i,:,:])

# # # # # # #   MAIN    # # # # # # # # #

if (__name__=='__main__'):

    #Creating an instance of the animation class will start the animation
    sim = interactive_animation(init_h,init_w,init_d,init_dt,n=init_n,I_factor=init_I)

    