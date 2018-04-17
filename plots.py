import matplotlib.pyplot as plt
import csv
import pickle

name = 'ErrorOnAcquisitions.csv'

err = []
with open(name) as csvfile:
	content = csv.reader(csvfile, delimiter=',')
	for row in content:
		err.append(float(row[1]))

pickle.dump(err, open('C:/Users/ecouturier/Desktop/head_tracker/results_elo/errors_raph.p', 'wb'))

fig = plt.figure()
plt.boxplot(err)
plt.xlabel('Methode 1 - Photoplethysmographie (2D)')
plt.ylabel('Erreur')
plt.title("Boite a moustaches de l'erreur sur les acquisitions\n de la base de donnees HCI Tagging avec la methode de detection a l'aide\n de photoplethysmographie sur image 2D")
fig.savefig('C:/Users/ecouturier/Desktop/head_tracker/results_elo/boxplot_raph.png')