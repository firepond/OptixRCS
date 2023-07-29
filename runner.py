import subprocess

model = "corner_reflector"
rcs_file = model + '_rcs.csv'
open(rcs_file, "w").close()

exePath = "C://development//optix//OptixRCS//build//bin//Release//TrianglesRcs.exe"
phi = 45
theta = 60
freq = 3E9
model = "d_reflector"
subprocess.run([exePath, str(phi), str(theta), str(freq), model])
print(model, phi, theta, freq)
