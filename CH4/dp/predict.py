import dpdata

training_systems = dpdata.LabeledSystem("training_data", fmt="deepmd/npy")
predict = training_systems.predict("train/graph.pb")

outfile = open("energy.txt","w")
a = training_systems["energies"]
b = predict["energies"]
for i in range(len(a)):
    print(a[i], b[i],file=outfile)

