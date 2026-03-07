import numpy as np
import matplotlib.pyplot as plt

def generate_dipole_signal(t_total=128, fs=1):
    """
    Simulates the magnetic anomaly signal of a target modeled as a magnetic dipole.
    Parameters match Table 3 of the DeepMAD paper.
    """
    # 1. Platform Settings
    v_kmh = 240
    v_ms = v_kmh * (1000.0 / 3600.0) # Convert to m/s
    
    # 2. Earth's Magnetic Field Vector (Normalized)
    u_earth = np.array([-0.025, 0.735, -0.677])
    u_earth = u_earth / np.linalg.norm(u_earth)
    
    # 3. Magnetic Moment (m)
    m_horizontal_amp = 100000.0
    m_vertical_amp = 300000.0
    # Randomize horizontal direction between 0 and 360 degrees
    theta = np.random.uniform(0, 2 * np.pi) 
    m = np.array([
        m_horizontal_amp * np.cos(theta),
        m_horizontal_amp * np.sin(theta),
        m_vertical_amp
    ])
    
    # 4. Closest Proximity Approach (CPA)
    r0 = np.random.uniform(300, 1000)
    
    # Time array (128 seconds at 1 Hz)
    t = np.arange(0, t_total, 1/fs)
    
    # Simulate a straight-line trajectory
    # Let's assume the sensor reaches CPA at exactly the halfway point (t=64s)
    t_cpa = t_total / 2.0
    
    # Relative coordinates of the sensor to the target over time
    x = v_ms * (t - t_cpa)
    y = np.full_like(t, r0) # Constant perpendicular distance (CPA)
    z = np.zeros_like(t)    # Assuming level altitude relative to target
    
    B_total = np.zeros_like(t)
    mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
    
    # Calculate the magnetic field at each second
    for i in range(len(t)):
        r_vec = np.array([x[i], y[i], z[i]])
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            continue # Prevent division by zero
            
        # Apply Equation 4: B(m,r)
        dot_product = np.dot(m, r_vec)
        term1 = 3 * (r_mag**-2) * dot_product * r_vec
        B_vec = (mu_0 / (4 * np.pi * r_mag**3)) * (term1 - m)
        
        # Apply Equation 5: B_total
        # Multiply by 1e9 to convert from Tesla to nanoTesla (nT)
        B_total[i] = np.dot(B_vec, u_earth) * 1e9
        
    return t, B_total

if __name__ == "__main__":
    print("Simulating magnetic dipole anomaly...")
    time_arr, signal = generate_dipole_signal()
    
    # Plotting to verify it matches Figure 7 from the paper
    plt.figure(figsize=(10, 5))
    plt.plot(time_arr, signal, color='#1f77b4', linewidth=2)
    plt.title("Simulated Magnetic Anomaly Signal")
    plt.xlabel("time (second)")
    plt.ylabel("magnetic field (nT)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()