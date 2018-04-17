import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

folder = 'C:/Users/ecouturier/Desktop/head_tracker/results_elo/pickle'
files = []

for dirnames, foldernames, filenames in os.walk(folder):
	files = filenames
	break

errors = []
for file in files:
	fname, _ = os.path.splitext(file)
	bpms,time,out,_ = pickle.load(open(folder + '/' + file, 'rb'))

	new_bpms = []
	new_time = []
	j = 0
	while bpms[j] == 0:
		j+=1
	for i,bpm in enumerate(bpms[j:]):
		if bpm != bpms[i-1]:
			new_bpms.append(bpm)
			new_time.append(time[i+j])
	bpms = new_bpms
	time = new_time

	mean = bpms[0]
	new_bpms = [mean]
	for i,bpm in enumerate(bpms):

		if len(new_bpms) > 10 and i > 10:
			mean = np.mean(new_bpms[i-10:i])
		else:
			mean = np.mean(new_bpms)

		if bpm-mean > 40:
			bpms[i] = bpm-abs(bpm-mean)*0.7

		elif bpm-mean < -40:
			bpms[i] = bpm+abs(bpm-mean)*0.7

		else:
			new_bpms.append(bpm)

	t = np.linspace(0,300,len(out[:300]))
	bpms = np.interp(t,time,bpms)
	time = t

	error = np.mean(abs(out[21:300]-bpms[21:]))

	fig = plt.figure()
	plt.plot(out[:300])
	plt.plot(bpms)
	plt.xlabel('Temps (sec)')
	plt.ylabel('BPM')
	plt.title('Erreur: ' + str(error))
	fig.savefig('C:/Users/ecouturier/Desktop/head_tracker/results_elo/graphs2/' + fname + '.png')

	print('\nError: ' + str(error) + '\n')
	errors.append(error)

pickle.dump(errors, open('C:/Users/ecouturier/Desktop/head_tracker/results_elo/errors.p', 'wb'))

fig = plt.figure()
plt.boxplot(errors)
plt.xlabel('Methode 2 - Analyse par composantes principales (2D)')
plt.ylabel('Erreur')
plt.title("Boite a moustaches de l'erreur sur les\n acquisitions de test a l'aide d'une analyse\n par composantes principales sur image 2D")
fig.savefig('C:/Users/ecouturier/Desktop/head_tracker/results_elo/boxplot.png')

