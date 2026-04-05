import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# ====================== PARAMETERS ======================
g = 9.8
dt = 0.001          # Smaller dt for better accuracy
t_max = 50          # Simulation time in seconds (you can increase)

# ==================== USER INPUTS ======================
m1 = float(input("Enter mass of pendulum 1 (kg): "))
m2 = float(input("Enter mass of pendulum 2 (kg): "))
L1 = float(input("Enter length of rod 1 (m): "))
L2 = float(input("Enter length of rod 2 (m): "))
theta1_deg = float(input("Enter initial angle for pendulum 1 (degrees): "))
theta2_deg = float(input("Enter initial angle for pendulum 2 (degrees): "))

theta1 = math.radians(theta1_deg)
theta2 = math.radians(theta2_deg)

z1 = 0.0   # initial angular velocity 1
z2 = 0.0   # initial angular velocity 2

# Perturbation for Lyapunov calculation
perturbation = 1e-6
theta2_pert = theta2 + perturbation

# ====================== DATA STORAGE ======================
t_arr = []

# Main system data
theta1_arr = []; theta2_arr = []
z1_arr = []; z2_arr = []
x1_arr = []; y1_arr = []
x2_arr = []; y2_arr = []

# Perturbed system (for divergence)
theta1_p_arr = []; theta2_p_arr = []
z1_p_arr = []; z2_p_arr = []

dist_arr = []       # phase space distance
log_dist_arr = []

# ====================== DERIVATIVES FUNCTION ======================
def derivatives(th1, z1, th2, z2, m1, m2, L1, L2):
    delta = th2 - th1
    den1 = (m1 + m2)*L1 - m2*L1*math.cos(delta)**2
    den2 = (L2 / L1) * den1
    
    dth1_dt = z1
    dz1_dt = (m2*L1*z1**2*math.sin(delta)*math.cos(delta) +
              m2*g*math.sin(th2)*math.cos(delta) +
              m2*L2*z2**2*math.sin(delta) -
              (m1+m2)*g*math.sin(th1)) / den1
    
    dth2_dt = z2
    dz2_dt = (-m2*L2*z2**2*math.sin(delta)*math.cos(delta) +
              (m1+m2)*g*math.sin(th1)*math.cos(delta) -
              (m1+m2)*L1*z1**2*math.sin(delta) -
              (m1+m2)*g*math.sin(th2)) / den2
    
    return dth1_dt, dz1_dt, dth2_dt, dz2_dt


# ====================== RK4 STEP ======================
def rk4_step(th1, z1, th2, z2, dt, m1, m2, L1, L2):
    k1t1, k1z1, k1t2, k1z2 = derivatives(th1, z1, th2, z2, m1, m2, L1, L2)
    k2t1, k2z1, k2t2, k2z2 = derivatives(th1 + 0.5*k1t1*dt, z1 + 0.5*k1z1*dt,
                                          th2 + 0.5*k1t2*dt, z2 + 0.5*k1z2*dt, m1, m2, L1, L2)
    k3t1, k3z1, k3t2, k3z2 = derivatives(th1 + 0.5*k2t1*dt, z1 + 0.5*k2z1*dt,
                                          th2 + 0.5*k2t2*dt, z2 + 0.5*k2z2*dt, m1, m2, L1, L2)
    k4t1, k4z1, k4t2, k4z2 = derivatives(th1 + k3t1*dt, z1 + k3z1*dt,
                                          th2 + k3t2*dt, z2 + k3z2*dt, m1, m2, L1, L2)
    
    th1_new = th1 + (dt/6)*(k1t1 + 2*k2t1 + 2*k3t1 + k4t1)
    z1_new  = z1  + (dt/6)*(k1z1 + 2*k2z1 + 2*k3z1 + k4z1)
    th2_new = th2 + (dt/6)*(k1t2 + 2*k2t2 + 2*k3t2 + k4t2)
    z2_new  = z2  + (dt/6)*(k1z2 + 2*k2z2 + 2*k3z2 + k4z2)
    
    return th1_new, z1_new, th2_new, z2_new


# ====================== SIMULATION ======================
n_steps = int(t_max / dt)
t = 0.0

# Initial conditions for both systems
th1, z1, th2, z2 = theta1, z1, theta2, z2
th1_p, z1_p, th2_p, z2_p = theta1, z1, theta2_pert, z2

for i in range(n_steps):
    # Advance both systems
    th1, z1, th2, z2 = rk4_step(th1, z1, th2, z2, dt, m1, m2, L1, L2)
    th1_p, z1_p, th2_p, z2_p = rk4_step(th1_p, z1_p, th2_p, z2_p, dt, m1, m2, L1, L2)
    
    t += dt
    t_arr.append(t)
    
    # Store main system data
    theta1_arr.append(th1)
    theta2_arr.append(th2)
    z1_arr.append(z1)
    z2_arr.append(z2)
    
    # Cartesian positions
    x1 = L1 * math.sin(th1)
    y1 = -L1 * math.cos(th1)
    x2 = x1 + L2 * math.sin(th2)
    y2 = y1 - L2 * math.cos(th2)
    
    x1_arr.append(x1)
    y1_arr.append(y1)
    x2_arr.append(x2)
    y2_arr.append(y2)
    
    # Phase space distance for Lyapunov
    dth1 = th1 - th1_p
    dz1  = z1  - z1_p
    dth2 = th2 - th2_p
    dz2  = z2  - z2_p
    distance = math.sqrt(dth1**2 + dz1**2 + dth2**2 + dz2**2)
    
    dist_arr.append(distance)
    log_dist_arr.append(math.log(distance + 1e-12))


# ====================== LYAPUNOV CALCULATION ======================
start_idx = int(0.2 * len(log_dist_arr))
end_idx   = int(0.7 * len(log_dist_arr))

slope, intercept, r_value, _, _ = linregress(t_arr[start_idx:end_idx], log_dist_arr[start_idx:end_idx])
lyapunov = slope

print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(f"Mass 1 (m1)     : {m1} kg")
print(f"Mass 2 (m2)     : {m2} kg")
print(f"Length 1 (L1)   : {L1} m")
print(f"Length 2 (L2)   : {L2} m")
print(f"Initial θ1      : {theta1_deg}°")
print(f"Initial θ2      : {theta2_deg}°")
print(f"Estimated Lyapunov Exponent (λ) : {lyapunov:.4f} s⁻¹")
print(f"R² of fit       : {r_value**2:.4f}")
print("="*60)


# ====================== PLOTTING ALL DATA ======================
fig = plt.figure(figsize=(14, 12))

# 1. Trajectory Plot
ax1 = plt.subplot(3, 2, 1)
ax1.plot(x1_arr, y1_arr, 'r', lw=1.2, label='Pendulum 1')
ax1.plot(x2_arr, y2_arr, 'b', lw=1.2, label='Pendulum 2')
ax1.set_aspect('equal')
ax1.set_title('Double Pendulum Trajectory')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Angles vs Time
ax2 = plt.subplot(3, 2, 2)
ax2.plot(t_arr, np.degrees(theta1_arr), 'r', label='θ₁ (degrees)')
ax2.plot(t_arr, np.degrees(theta2_arr), 'b', label='θ₂ (degrees)')
ax2.set_title('Angles vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (degrees)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Angular Velocities vs Time
ax3 = plt.subplot(3, 2, 3)
ax3.plot(t_arr, z1_arr, 'r', label='ω₁ (rad/s)')
ax3.plot(t_arr, z2_arr, 'b', label='ω₂ (rad/s)')
ax3.set_title('Angular Velocities vs Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angular Velocity (rad/s)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Log Distance (Lyapunov)
ax4 = plt.subplot(3, 2, 4)
ax4.plot(t_arr, log_dist_arr, 'g', lw=1.5, label='Actual separation')
ax4.plot(t_arr[start_idx:end_idx], 
         [intercept + slope*t for t in t_arr[start_idx:end_idx]], 
         'r--', lw=2, label=f'Fit: λ = {lyapunov:.4f} s⁻¹')
ax4.set_title('Log of Phase-Space Distance vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('ln(Distance)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. X Positions vs Time
ax5 = plt.subplot(3, 2, 5)
ax5.plot(t_arr, x1_arr, 'r', label='x₁')
ax5.plot(t_arr, x2_arr, 'b', label='x₂')
ax5.set_title('Horizontal Positions vs Time')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('X (m)')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
