import subprocess

model = "corner_reflector"
rcs_file = model + '_rcs.csv'
open(rcs_file, "w").close()

exePath = "C://development//optix//OptixRCS//build//bin//Release//TrianglesRcs.exe"
phi = "[0:90:1]"
# phi = ""
theta = "[60:60:1]"
# theta = ""
freq = "3E9"
model = "d_reflector"
subprocess.run([exePath, str(freq), str(phi), str(theta), model])
print(model, freq, phi, theta)
