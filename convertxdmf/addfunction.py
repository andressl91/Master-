out_file = []
count = 0

lines = open('./velocity.xdmf', 'r').readlines()
text = ""
for i, line in enumerate(lines):
    if i < 5:
        text += line
    elif '<Attribute Name=' in line:
        line = line.replace("f_19-1", "v")
        text += line
        text += lines[i+1]
        text += lines[i+2]

        text += line.replace('Name="v"', 'Name="d"')
        text += lines[i+1].replace("velocity.h5", "d.h5")
        text += lines[i+2]

    elif '<Attribute Name=' in lines[i-1] or '<Attribute Name=' in lines[i-2]:
        continue

    else:
        text += line

f = open("./ordered2.xdmf", "w")
f.write(text)
f.close()
