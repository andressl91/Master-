import os
#LAGRANGE MULTIP
file_vel = "2D_vel"
file = open(file_vel, "w"); file.seek(0)
file.truncate(); file.close

#2D CASE
#os.system("cd 2D")
#os.system("python 2D/taylorg.py")
os.system("python /home/andreas/Desktop/Oasis/NSfracStep.py problem=TaylorGreen2D")
######
#CONTINUE ON OASIS 2D CASE BOTTOM, TO GET L2 norm
