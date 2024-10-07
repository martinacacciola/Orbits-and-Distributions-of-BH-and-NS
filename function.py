import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import os
from fireworks.particles import Particles
from scipy.stats import gaussian_kde
from fireworks.nbodylib.nunits import Nbody_units
from fireworks.ic import ic_two_body
from fireworks.nbodylib.dynamics import *
from fireworks.nbodylib.integrators import *
from sevnpy.sevn import SEVNmanager, Star
from fireworks.nbodylib.potentials import MultiPotential, LogHalo, MyamotoNagai, Hernquist, NFW, TruncatedPlaw
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

# SEVN Initialization Function

# Call this function to initialize the parameters of SEVN
Hobbs = {"sn_kicks":"hobbs"}
Hobbs_pure = {"sn_kicks":"hobbs_pure"}
fast = {"sn_kick_velocity_stdev":265}
slow = {"sn_kick_velocity_stdev":150}

# Define a function that takes two arguments: kick_model and kick_speed
def sevn_init(kick_model, kick_speed):
    # Check if the kick_model is valid
    if kick_model not in [Hobbs, Hobbs_pure]:
        print("Invalid kick model. Please choose either Hobbs or Hobbs_pure.")
        return
    # Check if the kick_speed is valid
    if kick_speed not in [fast, slow]:
        print("Invalid kick speed. Please choose either fast or slow.")
        return
    # Set the parameters for the SEVN manager based on the arguments
    params = {**kick_model, **kick_speed}
    
    SEVNmanager.init(params)
    # Return the SEVN manager object
    return SEVNmanager

# Units and NU i nitialization
## 1-Nbody units scaling

# Select the proper scaling for the MW, Nbody_units works by default with Msun, pc, km/s, Myr, so 
##  NU = Nbody_units(M=1,L=1000,T=1000) # the physical inputs are assumed in units of 1 Msun, kpc, km/s, Gyr

Mscale= 1e10 # Use a mass scale of 10^10 Msun
rscale = 1000 # Use a length scale of 1 kpc
Tscale = 1000 # Use a time scale of 1 Gyr

NU = Nbody_units(M=Mscale,L=rscale, T=Tscale) # We can now use nu to rescale the potential and the particles position from physical to Nbody 

# - DMhalo 
d=12 # kpc
vf=212 # km/s
dn=NU.pos_to_Nbody(d) # In Nbody
vfn=NU.vel_to_Nbody(vf) # in Nbody
halo = LogHalo(vflat=vfn,a=dn) # Set the LogHalo potential with the proper Nbody units 

# Hernquist spheroid/bulge
c=0.7 # kpc
Mspher=3.4 # Msun in units of 1e10
cn=NU.pos_to_Nbody(c)
Msphern=NU.m_to_Nbody(Mspher)
bulge=Hernquist(Mass=Msphern,a=cn)

# Myamoto-Nagai disc 
a=6.5 # kpc
b=0.260 # kpc
Mdisc=10 # Msun in units of 1e10
an=NU.pos_to_Nbody(a)
bn=NU.pos_to_Nbody(b)
Mdiscn=NU.m_to_Nbody(Mdisc)
disc=MyamotoNagai(Mass=Mdiscn,a=an,b=bn)

# Finally use the MultiPotential method  to set the object thatn can be passed to the integrator 
MWJ95=MultiPotential([halo,disc,bulge])

# CDF for R
def density_profile_R(x):
    return 1 - np.exp(-x) * (1 + x)

# CDF for z
def density_profile_z(x):
    return 1 - np.exp(-np.abs(x))

# interpolate the CDF using the the rescaled cdf values and the original grid values
# So R,z can be sampled interpolating the cdf and using the inverse sampling

def interpolate_draw(xmin, xmax, f, r_or_z_or_m = "m"):
    
    x = np.linspace(xmin, xmax, 100)

    # Evaluate the cdf
    u = f(x)
    
    #rescaling
    u = (u - u[0])/(u[-1]-u[0])

    # Interpolate
    fint = scipy.interpolate.interp1d(u, x)

    # Draw from F(x) using inverse sampling
    u_draw = np.random.uniform(0, 1)

    R_d = 2.6
    z_d = 0.3

    if r_or_z_or_m == "R":
        R_draw = fint(u_draw)*R_d
        return R_draw

    if r_or_z_or_m == "z":
        z_draw = fint(u_draw)*z_d
        return z_draw
    
    if r_or_z_or_m == "m":
        M_draw = fint(u_draw)
        return (M_draw * 1e-10)

def cylindrical_to_cartesian(R, phi, z):
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = z

    return x, y, z

## Mass - Kroupa IMF

# Draw the initial stellar mass using a Kroupa IMF (Kroupa et al., 2001, see Lecture 7) 
# from Mmin = 9 solar masses M to Mmax = 150 solar masses M. Since we are interested in 
# compact remnants, we can neglect stars with M < 9 solar masses M

# Kroupa IMF parameters
M_min = 9.0 # Minimum stellar mass
M_max = 150.0 # Maximum stellar mass

K = 1 / (np.log(M_max) - np.log(M_min))

def kroupa_imf(m):
    return K * m**(-2.35)

## v_R, v_z, v_φ - velocities

# For the initial velocity, you can put your stars in nearly-circular orbits. In cylindrical coordinates, 
# it is possible to extract velocities from Gaussians: vR ∼ N(0,σR), vz ∼ N(0,σz), vphi ∼ N(Vcirc(R),σphi) 
# where Vcirc(R), = np.sqrt(R*aR), where aRis the total acceleration due to the MW potential at the position . 
# For thin disk population, σR = 36 km/s σϕ = 24 km/s σz = 16 km/s


def cylindrical_velocities_to_cartesian(vR, vphi, vz, phi):
    # Convert cylindrical velocities to Cartesian velocities
    vx = vR * np.cos(phi) - vphi * np.sin(phi)
    vy = vR * np.sin(phi) + vphi * np.cos(phi)
    vz = vz
    
    return vx, vy, vz

# Define parameterseR)
sigma_R = 36.0
sigma_Rn = NU.vel_to_Nbody(sigma_R)
sigma_phi = 24.0
sigma_phin = NU.vel_to_Nbody(sigma_phi)
sigma_z = 16.0
sigma_zn = NU.vel_to_Nbody(sigma_z)

## Generate IC: particle with pos, vel, mass

def generate_position_velocity_mass_arrays(num_samples):
    
    positions = []
    velocities = []
    masses = []

    potentials = []

    for _ in range(num_samples):

        # Sample Positions
        # CHECK THE RANGES FOR R AND z
        uR_draw = interpolate_draw(0, 1, density_profile_R, r_or_z_or_m = "R")
        uz_draw = interpolate_draw(0, 1, density_profile_z, r_or_z_or_m = "z")
        R = uR_draw
        sign_func = lambda u_sign: np.sign(1) if u_sign < 0.5 else np.sign(-1)
        u_sign_value = np.random.uniform(0, 1)
        z_sign = sign_func(u_sign_value)
        z = z_sign * uz_draw
        phi = np.random.uniform(0, 2 * np.pi)

        R = NU.pos_to_Nbody(R)
        z = NU.pos_to_Nbody(z)

        # Sample Mass in NBody Units
        uM_draw = interpolate_draw(M_min, M_max, kroupa_imf, r_or_z_or_m = "m")
        M = NU.m_to_Nbody(uM_draw)

        masses.append(M)

        # Sample Velocities in NBody Units
        vR = np.random.normal(0, sigma_Rn)
        vz = np.random.normal(0, sigma_zn)

        # Used np.abs in case we get a negative a_r
        a_r = np.abs(MWJ95.evaluate(R, z)[0][:,0])
        potentials.append(MWJ95.evaluate(R, z)[2][0][0])
        V_circ_R = np.sqrt(R * a_r)

        vphi = np.random.normal(V_circ_R, sigma_phin)

        # Convert to Cartesian coordinates and NBody Units
        x, y, z = cylindrical_to_cartesian(R, phi, z)

        positions.append([x, y, z])

        # Convert to Cartesian coordinates and NBody Units
        vx, vy, vz = cylindrical_velocities_to_cartesian(vR, vphi, vz, phi)

        velocities.append([vx[0], vy[0], vz])

        # Create the particle object
        part = Particles(position=positions, velocity=velocities, mass=masses)

    return part, potentials


## Evolution

# Stellar evolution with SEVNpy, to estimate the stellar lifetime that sets the
# first part of the orbit integration you will use SEVNpy (Lecture 9, see the
# SEVNpy documentation here). Evolve the stars until the formation of a
# remnant to get the lifetime.

## Evolution

# Stellar evolution with SEVNpy, to estimate the stellar lifetime that sets the
# first part of the orbit integration you will use SEVNpy (Lecture 9, see the
# SEVNpy documentation here). Evolve the stars until the formation of a
# remnant to get the lifetime.

def evolve_stars(num_samples, Z, t_evo):

    # Evolve stars for t_evo lenth

    # Generate particles positions, velocities and masses with sampling
    particle, _ = generate_position_velocity_mass_arrays(num_samples)
    # Pos in NBody units
    pos_nb = particle.pos
    # Vel in NBody units
    vel_nb = particle.vel
    # Mass in NBody units
    mass_nb = particle.mass
    # Pos in MSun units
    mass = NU.m_to_Msun(particle.mass)

    # Initialize Particles
    part = Particles(position=pos_nb, velocity=vel_nb, mass=mass_nb)

    # Initialize empty list of v_kick
    vkick = []

    # Create a star list for each of the bodies in our simulation
    # Change the random seed in stards initialization for i,mass enumerate(mass) rseed = i
    stars_list = [Star(Mzams=massi, rseed=index, Z=Z, tini="cheb") for index, massi in enumerate(mass)]
    # Set a simulation time for the system
    print("System evolution:", t_evo, "Gyr")

    # Iterate over every star
    for i in range(len(stars_list)):

        # Evolve star and update the particle's mass
        stars_list[i].evolve(tend=t_evo)
        vkick.append(stars_list[i].get_SN_kick()["Vkick"])

    return vkick, stars_list, part

def data_df(vkick, stars_list):
    
    # Initialize dataframe with x,y,z components of v_kicks
    df = pd.DataFrame(vkick, columns=['vx_kick', 'vy_kick', 'vz_kick'])

    # Initialize an empty list to store data
    data = []

    # Iterate over each sub_df in stars_list corresponding to a star
    for sub_df in stars_list:
        # Get the time and mass of the star at SN phase shift
        second_to_last_row = sub_df.getp().iloc[-2]  
        localtime = second_to_last_row['Localtime']
        mass = second_to_last_row['Mass']
        remnant_type = second_to_last_row['RemnantType']
        
        # Repeat localtime and mass 1000 times
        data.append([localtime, mass, remnant_type])


    # Create a DataFrame from the data
    second_df = pd.DataFrame(data, columns=['Time', 'Mass', 'Remnant Type'])

    # Merge Dataframe
    merged_df = pd.merge(df, second_df, left_index=True, right_index=True)

    return merged_df

def integration_loop(df, initial_part, Tend_nb, dt_nb, skip_step_after_kick_phase=100):
    
    # Orbital integration loop

    # Initialize t at 0
    t = 0

    # Initialize positions, velocities, times, masses and number of stars lists to save the data at each iteration
    pos = []
    velocities = []
    tl = []
    ml = []
    num_stars = []
    time_at_kick = []

    ns_idx = []
    ns_expelled = 0

    bh_idx = []
    bh_expelled = 0

    tot_expelled = 0

    # Initialize the starting number of stars
    n_stars = 0

    # Initialize a set to keep track of indices where kick is applied
    applied_kick_indices = set()

    # Compute the specific time values to check
    specific_times = set(int(np.max(df['Time'])) + np.arange(int(np.max(df['Time'])), Tend_nb, skip_step_after_kick_phase))

    # Evolution
    while t <= Tend_nb:

        # Save the pos, vel, mass, times, n_stars at beginning of evolution
        if t == 0:

            print("Time:", t)
            pos.append(initial_part.pos)
            velocities.append(initial_part.vel)
            ml.append(initial_part.mass.copy())
            tl.append(t)

        # Integrate the particles at beginning of evolution before the kick
        if t > 0 and t < np.min(df['Time']):

            print("Time before kick:", t)

            # Integrate particles
            final_part, _, _, _, _ = integrator_verlet(particles=initial_part, acceleration_estimator=MWJ95.acceleration, tstep=dt_nb)
            
            # Append integrated pos, vel, mass, t
            pos.append(final_part.pos)
            velocities.append(final_part.vel)
            ml.append(final_part.mass)
            tl.append(t)

        # Integrate the particles at the kick
        if t >= np.min(df['Time']) and t <= np.max(df['Time']):

            print("Time at kick:", t)

            # Iterate through each row in the DataFrame
            for index in range(len(initial_part.pos)):

                t = round(t, 4 - len(str(int(t))))
                t_df = round(df['Time'][index], 4 - len(str(int(df['Time'][index]))))

                if t == t_df:

                    # Temporary velocity and mass arrays to save v_kicks and m_kicks for each star
                    vel = []
                    
                    # Check if current star has been kicked already
                    if index not in applied_kick_indices:

                        if index >= len(initial_part.pos):
                            pass

                        else:
                            # Get the x,y,z v_kick component from df
                            vel_kick = [df['vx_kick'][index], df['vy_kick'][index], df['vz_kick'][index]]
                            
                            # Apply the kick and save the velocity
                            vel = (list(vel_kick + initial_part.vel[index]))
                            initial_part.vel[index] = vel
                            applied_kick_indices.add(index)  # Add index to set of applied kicks

                            # Convert Mass kick to NBody units
                            mass_kick = df['Mass'][index] * 1e-10
                            initial_part.mass[index] = mass_kick

                            x, y, z = initial_part.pos[:, 0], initial_part.pos[:, 1], initial_part.pos[:, 2]
                            R = np.sqrt(x**2 + y**2)

                            # Calculate Potential energy of star felt by MW potential
                            U_star = MWJ95.evaluate(R, z)[0][:,0][index]

                            # Calculate the magnitude of the velocity vector
                            velocity_magnitude = np.linalg.norm(initial_part.vel[index])

                            # Calculate kinetic energy
                            KE_star = 0.5 * mass_kick * velocity_magnitude**2

                            # Calculate total star energy after kick
                            Energy = U_star + KE_star

                            # Remove the star if its energy is > 0, else keep it

                            if np.abs(Energy) > 0:

                                initial_part.pos = np.delete(initial_part.pos, index, axis=0)
                                initial_part.vel = np.delete(initial_part.vel, index, axis=0)
                                initial_part.mass = np.delete(initial_part.mass, index, axis=0)

                                if df['Remnant Type'][index] == 5:
                                    ns_expelled += 1

                                if df['Remnant Type'][index] == 6:
                                    bh_expelled += 1

                                tot_expelled += 1
                                
                            else:
                                pass

                            # Integrate particles
                            final_part, _, _, _, _ = integrator_verlet(particles=initial_part, acceleration_estimator=MWJ95.acceleration, tstep=dt_nb)
                            
                            # Append integrated pos, vel, mass, t
                            pos.append(final_part.pos)
                            velocities.append(final_part.vel)
                            ml.append(final_part.mass)
                            tl.append(t)
                            n_stars += 1
                            num_stars.append(n_stars)
                            time_at_kick.append(t)

        # Integrate the stars at periods of skip_step_after_kick_phase
        if t > np.max(df['Time']):
            if int(t) in specific_times:
                
                print("Time after kick:", t)
                final_part, _, _, _, _ = integrator_verlet(particles=initial_part, acceleration_estimator=MWJ95.acceleration, tstep=dt_nb)

                # Append integrated pos, vel, mass, t
                pos.append(final_part.pos)
                velocities.append(final_part.vel)
                ml.append(final_part.mass)
                tl.append(t)

        # Save the remnant type index for each star
        if round(t, 4 - len(str(int(t)))) == round(Tend_nb, 4 - len(str(int(Tend_nb)))):
            
            for index in range(len(initial_part.pos)):

                if df['Remnant Type'][index] == 5:
                    ns_idx.append(index)

                if df['Remnant Type'][index] == 6:
                    bh_idx.append(index)
                    
                else:
                    pass

        # Update time
        t += dt_nb

    return pos, velocities, ml, tl, num_stars, time_at_kick, ns_idx, bh_idx, ns_expelled, bh_expelled, tot_expelled

def plot_remnant_expelled(ns_expelled, bh_expelled, tot_expelled):

    # Data
    categories = ['BH Remnant expelled%', 'NS Remnant expelled %']
    counts = [bh_expelled/tot_expelled * 100, ns_expelled/tot_expelled * 100]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=['black', 'gray'])
    plt.xlabel('Categories')
    plt.ylabel('Percentage')
    plt.show()

def plot_evolution_pos(pos, ns_idx, bh_idx):

    initial_x = pos[0][:, 0]
    initial_y = pos[0][:, 1]
    final_x = pos[-1][:, 0]
    final_y = pos[-1][:, 1]
    
    plt.plot(initial_x, initial_y, 's', color='r', label='Initial Position', markersize=2)

    # Plot neutron stars and black holes separately for final positions
    plt.plot(final_x[ns_idx], final_y[ns_idx], 's', color='grey', label='Neutron Stars (Final)', markersize=2)
    plt.plot(final_x[bh_idx], final_y[bh_idx], 's', color='black', label='Black Holes (Final)', markersize=2)

    # Calculate mean and standard deviation for x and y coordinates
    initial_mean_x = np.mean(initial_x)
    initial_std_x = np.std(initial_x)
    initial_mean_y = np.mean(initial_y)
    initial_std_y = np.std(initial_y)
    
    final_mean_x = np.mean(final_x)
    final_std_x = np.std(final_x)
    final_mean_y = np.mean(final_y)
    final_std_y = np.std(final_y)

    final_mean_bh_x = np.mean(final_x[bh_idx])
    final_std_bh_x = np.std(final_x[bh_idx])
    final_mean_bh_y = np.mean(final_y[bh_idx])
    final_std_bh_y = np.std(final_y[bh_idx])

    final_mean_ns_x = np.mean(final_x[ns_idx])
    final_std_ns_x = np.std(final_x[ns_idx])
    final_mean_ns_y = np.mean(final_y[ns_idx])
    final_std_ns_y = np.std(final_y[ns_idx])

    # Plot mean points with standard deviation error bars for initial position
    plt.errorbar(initial_mean_x, initial_mean_y, xerr=initial_std_x, yerr=initial_std_y, fmt='s', color='green', label='Initial Mean ± Std', markersize=4)

    # Plot mean points with standard deviation error bars for final position
    plt.errorbar(final_mean_x, final_mean_y, xerr=final_std_x, yerr=final_std_y, fmt='o', color='blue', label='Final Mean ± Std', markersize=1)

    # Plot mean points with standard deviation error bars for final position
    plt.errorbar(final_mean_bh_x, final_mean_bh_y, xerr=final_std_bh_x, yerr=final_std_bh_y, fmt='o', color='black', label='BH Final Mean ± Std', markersize=4)

    # Plot mean points with standard deviation error bars for final position
    plt.errorbar(final_mean_ns_x, final_mean_ns_y, xerr=final_std_ns_x, yerr=final_std_ns_y, fmt='o', color='grey', label='NS Final Mean ± Std', markersize=4)


    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel("x [Nbody]", fontsize=16)
    plt.ylabel("y [Nbody]", fontsize=16) 
    plt.tight_layout()
    plt.title('Final position on the disk plane')
    plt.legend(fontsize=10)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

def plot_evolution_r(pos, ns_idx, bh_idx):
    zi = pos[0][:, 2]
    z = pos[-1][:, 2]
    r = np.sqrt((pos[-1][:, 0]**2 + pos[-1][:, 1]**2))
    ri = np.sqrt((pos[0][:, 0]**2 + pos[0][:, 1]**2))
    
    plt.plot(ri, zi, 's', color='r', label=f'Initial Position', markersize=2)
    plt.plot(r[ns_idx], z[ns_idx], 's', color='grey', label='Neutron Stars (Final)', markersize=2)
    plt.plot(r[bh_idx], z[bh_idx], 's', color='black', label='Black Holes (Final)', markersize=2)
    

    # Plotting mean points with standard deviation error bars
    plt.errorbar(np.mean(ri), np.mean(zi), xerr=np.std(ri), yerr=np.std(zi), fmt='s', color='green', label='Initial Mean ± Std', markersize=4)
    plt.errorbar(np.mean(r), np.mean(z), xerr=np.std(r), yerr=np.std(z), fmt='o', color='blue', label='Final Mean ± Std', markersize=1)
    plt.errorbar(np.mean(r[bh_idx]), np.mean(z[bh_idx]), xerr=np.std(r[bh_idx]), yerr=np.std(z[bh_idx]), fmt='o', color='black', label='BH Final Mean ± Std', markersize=4)
    plt.errorbar(np.mean(r[ns_idx]), np.mean(z[ns_idx]), xerr=np.std(r[ns_idx]), yerr=np.std(z[ns_idx]), fmt='o', color='grey', label='NS Final Mean ± Std', markersize=4)

    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel("r (distance from galactic centre) [Nbody]", fontsize=16)
    plt.ylabel("z (height wrt the galactic plane) [Nbody]", fontsize=16)
    plt.xscale('log')
    plt.tight_layout()
    plt.title('Height-Radius wrt center of the galaxy')
    plt.legend(fontsize=10)
    plt.xlim(-10, 10)
    plt.ylim(-0.4, 0.4)
    plt.show()

def plot_evolution_mass(tl, ml, time_last_kick):
    for i in range(len(tl)):
        if tl[i] < time_last_kick:
            sublist = ml[i]
            sum_m = np.sum(sublist)
            std_dev = np.std(sublist)
            plt.errorbar(tl[i], sum_m, yerr=std_dev, fmt='o', color='black', capsize=2, label='Average Mass ± Std', markersize=2)
        else:
            pass

    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel("t [Nbody]", fontsize=16)
    plt.ylabel("Mass [Nbody]", fontsize=16)
    plt.tight_layout()
    plt.title('Mass evolution')
    plt.show()

def plot_number_stars_evolution(times, num_stars):
    """
    Plot the number of stars evolving through time.

    Parameters:
    times (list): List of time points.
    num_stars (list): List of the number of stars corresponding to each time point.
    """
    plt.plot(times[:len(num_stars)], num_stars, marker='o', linestyle='-', color='black', markersize=2)
    plt.xlabel('t [Nbody]')
    plt.ylabel('Number of Stars')
    plt.title('Number of Stars Kicked')
    plt.grid(True)
    plt.show()

def plot_path_through_time(pos_array):
    """
    Plot the path through all time for each point.

    Parameters:
    pos_array (numpy.ndarray): Array of positions in 2D space, shape (N, T, 2),
                               where N is the number of points and T is the number of time steps.
    """
    num_points = len(pos_array)
    # Plot path for each point with consistent color
    for i in range(num_points):
        x = pos_array[i][:, 0]
        y = pos_array[i][:, 1]
        plt.scatter(x, y, label=f'Point {i+1}', color='black', s=0.00001)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title('Path of Points Through Time')
    plt.grid(True)
    plt.show()

def plot_path_through_time_3d(pos_array):
    """
    Plot the path through all time for each point in 3D space.

    Parameters:
    pos_array (numpy.ndarray): Array of positions in 3D space, shape (T, N, 3),
                               where N is the number of points and T is the number of time steps.
    """
    num_points = len(pos_array)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot path for each point with a different color
    for i in range(num_points):
        x = pos_array[i][:, 0]
        y = pos_array[i][:, 1]
        z = pos_array[i][:, 2]
        ax.scatter(x, y, z, label=f'Point {i+1}', c='black', marker='o', s=0.000001)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title('Path of Points Through Time')
    plt.grid(True)
    plt.show()
    
def plots(pos, tl, ml, num_stars, time_at_kick, time_last_kick, ns_idx, bh_idx, ns_expelled, bh_expelled, tot_expelled):
    plot_remnant_expelled(ns_expelled, bh_expelled, tot_expelled)
    plot_evolution_pos(pos, ns_idx, bh_idx)
    plot_evolution_r(pos, ns_idx, bh_idx)
    plot_evolution_mass(tl, ml, time_last_kick)
    plot_number_stars_evolution(time_at_kick, num_stars)
    plot_path_through_time(pos)
    plot_path_through_time_3d(pos)

def plot_evolution_mass_total(tls, mls, labels):
    
    plt.figure(figsize=(10, 6))  # Adjust figure size if necessary
    
    for tl, ml, label in zip(tls, mls, labels):
        mean_list = []
        stdev_list = []

        for i, sublist in enumerate(ml):
            mean = np.mean(sublist)
            std_dev = np.std(sublist)
            mean_list.append(mean)
            stdev_list.append(std_dev)
        
        plt.errorbar(tl[::20], mean_list[::20], yerr=stdev_list[::20], fmt='o', capsize=5, label=label)

    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel("t [Nbody]", fontsize=16)
    plt.ylabel("Mass [Nbody]", fontsize=16)
    plt.tight_layout()
    plt.title('Mass evolution')
    plt.legend(fontsize=6)
    plt.show()

def plot_velocity_distribution_total(velocities_list, labels_list):

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['vx', 'vy', 'vz']

    for i in range(3):
        for idx, velocities in enumerate(velocities_list):
                
            # Calculate average velocities
            avg_velocities = np.mean(velocities_list[idx], axis=0)
            all_velocities_axis = avg_velocities[:, i]
            kde = gaussian_kde(all_velocities_axis)
            x_vals = np.linspace(min(all_velocities_axis), max(all_velocities_axis), 100)
            density = kde(x_vals)
            axes[i].plot(x_vals, density, label=labels_list[idx])
        
        axes[i].set_xlabel(labels[i] + ' [Nbody]')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{labels[i]} Distribution')
        axes[i].legend()  # Add legend
        
    plt.tight_layout()
    plt.show()

def plot_number_stars_evolution_total(times, num_stars_list, labels_list):
    """
    Plot the number of stars evolving through time.

    Parameters:
    times (list): List of time points.
    num_stars_list (list of lists): List of lists of the number of stars corresponding to each time point.
    labels_list (list): List of labels corresponding to each dataset.
    """
    for num_stars, label in zip(num_stars_list, labels_list):
        plt.plot(times, num_stars, marker='o', linestyle='-', label=label)

    plt.xlabel('Time')
    plt.ylabel('Number of Stars')
    plt.title('Number of Stars Evolution')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_evolution_r_total(pos_list, labels_list):
    """
    Plot the height-radius wrt center of the galaxy.

    Parameters:
    pos_list (list of arrays): List of arrays containing positions.
    labels_list (list): List of labels corresponding to each dataset.
    """
    _, ax = plt.subplots(figsize=(8, 6))
    
    for positions, label in zip(pos_list, labels_list):
        # initial
        zi = positions[0][:, 2]
        ri = np.sqrt((positions[0][:, 0]**2 + positions[0][:, 1]**2))

        # final
        z = positions[-1][:, 2]
        r = np.sqrt((positions[-1][:, 0]**2 + positions[-1][:, 1]**2))

        # Plotting mean points with standard deviation error bars
        ax.errorbar(np.mean(ri), np.mean(zi), xerr=np.std(ri), yerr=np.std(zi), fmt='s', label=f'{label} | Initial Mean ± Std')
        ax.errorbar(np.mean(r), np.mean(z), xerr=np.std(r), yerr=np.std(z), fmt='o', label=f'{label} | Final Mean ± Std')

    ax.tick_params(axis='both', which='major', labelsize=16)  # Set tick label size
    ax.set_xlabel("r (distance from galactic centre) [Nbody]", fontsize=16)
    ax.set_ylabel("z (height wrt the galactic plane) [Nbody]", fontsize=16)
    ax.set_xscale('log')
    ax.set_title('Height-Radius wrt center of the galaxy', fontsize=16)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_particle_path_3D(times_list, particle_list, animation_name):
    """
    Animates the 3D path of particles over time.

    Parameters:
    - times_list (list): List of time values corresponding to each frame.
    - particle_list (list): List of particle positions for each frame.
    - animation_name (str): Name of the output animation file.

    Returns:
    - Animation: Displays the 3D animation.
    """

    def update(frame):

        print(f"Processing frame {frame + 1}/{len(times_list)}")
            
        ax.clear()
        ax.set_title(f'Time: {times_list[frame]}', color='white')

        for i, particles in enumerate(particle_list[frame]):
            x, y, z = particles
            ax.scatter(x, y, z, marker='o', c='white', label='Particle', s=0.1)  # Set particle size to 10

            if frame == 0:
                particle_paths[i] = np.array([[x, y, z]])
            else:
                particle_paths[i] = np.concatenate([particle_paths[i], [[x, y, z]]])

        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')

        # Set limits centered at (0, 0, 0)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)

        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = 'black'
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='black')  # Set grid color to black

    fig = plt.figure()
    fig.set_facecolor('black')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    particle_paths = [np.array([]) for _ in range(len(particle_list[0]))]
    
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = 'black'
    ax.xaxis.pane.linecolor = ax.yaxis.pane.linecolor = ax.zaxis.pane.linecolor = 'black'

    # Calculate fps to make animation last 5 seconds
    fps = len(times_list) / 5

    animation = FuncAnimation(fig, update, frames=len(times_list), interval=20, repeat=False)

    # Save the animation as a GIF
    animation.save(animation_name, writer='pillow', fps=fps)

def calculate_velocity_magnitude(v_x, v_y, v_z):
    """
    Calculate the magnitude of the velocity given its x, y, and z components.

    Parameters:
    - vx (float): Velocity component in the x-direction.
    - vy (float): Velocity component in the y-direction.
    - vz (float): Velocity component in the z-direction.

    Returns:
    - float: Magnitude of the velocity.
    """
    return np.sqrt(v_x**2 + v_y**2 + v_z**2)

def plot_distributions_with_kde(data1, data2, labels, attribute):
    """
    Plot the distributions of multiple datasets with smoothed KDE curves and means.

    Parameters:
    - data1 (list of array-like): List of data arrays for black holes to be plotted.
    - data2 (list of array-like): List of data arrays for neutron stars to be plotted.
    - labels (list of str): List of labels for the datasets.
    - attribute (str): Name of the attribute being plotted.

    Returns:
    - None
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    for i, dataset in enumerate(data1):
        # Smoothed KDE curve with different shades of blue
        sns.kdeplot(dataset, color=f'C{i}', linestyle='-', linewidth=2, label=labels[i])
        # Mean of the dataset
        mean_value = np.mean(dataset)
        plt.axvline(mean_value, color=f'C{i}', linestyle='--', label=f'Mean {labels[i]}: {mean_value:.2f}')
    
    plt.title(f'BH Distributions of {attribute} Smoothed KDE Curves with Mean')
    plt.xlabel(f'{attribute} [Nbody]')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    for i, dataset in enumerate(data2):
        # Smoothed KDE curve with different shades of blue
        sns.kdeplot(dataset, color=f'C{i}', linestyle='-', linewidth=2, label=labels[i])
        # Mean of the dataset
        mean_value = np.mean(dataset)
        plt.axvline(mean_value, color=f'C{i}', linestyle='--', label=f'Mean {labels[i]}: {mean_value:.2f}')
    
    plt.title(f'NS Distributions of {attribute} Smoothed KDE Curves with Mean')
    plt.xlabel(f'{attribute} [Nbody]')
    plt.ylabel('Density')
    plt.legend(fontsize = 8)
    plt.show()