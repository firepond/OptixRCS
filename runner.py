import subprocess

# model = "corner_reflector"
model = "d_reflector"
rcs_file = "output/" + model + '_rcs.csv'
exePath = "C://development//optix//OptixRCS//build//bin//Debug//TrianglesRcs.exe"


phi = "90,90,1"
theta = "90,90,1"
freq = "15E9"
subprocess.run([exePath, str(freq), str(phi), str(theta), model])
print(model, freq, phi, theta)
