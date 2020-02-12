import numpy as np
import copy 
from numpy import loadtxt
import time
import matplotlib.pyplot as plt
#del a[i] i indice
# a.pop(i) i indice
# a.remove(elemento) 

distancias = loadtxt('road_distance.txt')

'''
distancias=[[0,12,3,23,1,5,23,56,12,11,89,97,52],
            [12,0,9,18,3,41,45,5,41,27,16,76,56],
             [3,9,0,89,56,21,12,48,14,29,5,91,8],
             [23,18,89,0,87,46,75,17,50,42,100,70,15],
             [1,3,56,87,0,55,22,86,14,33,31,84,21],
             [5,41,21,46,55,0,21,76,54,81,92,37,22],
             [23,45,12,75,22,21,0,11,57,48,39,59,22],
             [56,5,48,17,86,76,11,0,63,24,55,58,98],
             [12,41,14,50,14,54,57,63,0,9,44,18,52],
             [11,27,29,42,33,81,48,24,9,0,64,65,82],		
	     [89,16,5,100,31,92,39,55,44,64,0,9,70],
	     [97,76,91,70,84,37,59,58,18,65,9,0,50],
             [52,56,8,15,21,22,22,98,52,82,70,50,0]]

'''
nodos=[]
for i in range(226):  #226
	nodos.append(i)


def m_visibilidad():
	size=len(distancias)
	visibilidad=[0]*size
	for i in range(size):
		visibilidad[i]=[0]*size
		for j in range(size):
			if i!=j:
				visibilidad[i][j]=float("%.5f"%(1.0/distancias[i][j]))
			else:
				visibilidad[i][j]=0

	return visibilidad


def m_feromonas():
	size=len(distancias)
	feromonas=[0]*size
	for i in range(size):
		feromonas[i]=[0]*size
		for j in range(size):
			if i!=j:
				feromonas[i][j]=feromona_inicial
			else:
				feromonas[i][j]=0

	return feromonas


def fitness(ruta):
	costo=0
	size=len(ruta)-1
	for i in range(size):
		costo+=distancias[ruta[i]][ruta[i+1]]	

	return costo

def diversificacion(permutacion,permutacion_i,valor_inicial,condicion,final):
	valor_temp=valor_inicial
	suma=0.0		
	size=len(permutacion)
	probabilidad=[None]*size
	for i in range(size):
		t=feromona[valor_inicial][permutacion[i]]
		t=pow(t,alpha) 
		n=visibilidad[valor_inicial][permutacion[i]] 
		n=pow(n,beta) 
		probabilidad[i]=t*n	
		suma+=t*n		
	
	suma_probabilidad=0.0
	for i in range(size):
		if suma==0.0:		
			probabilidad[i]=float("%.5f"%(((1.0)*probabilidad[i]/0.01)))
			suma_probabilidad+=probabilidad[i]
			probabilidad[i]=suma_probabilidad
		else:
			probabilidad[i]=float("%.5f"%(((1.0)*probabilidad[i]/suma)))
			suma_probabilidad+=probabilidad[i]
			probabilidad[i]=suma_probabilidad
	
	aleatorio=np.random.uniform(0,1)
	
	for i in range(size):
		if aleatorio<probabilidad[i]:
			valor_inicial=permutacion[i]
			permutacion_i.append(valor_inicial)
			if valor_inicial==final:
				condicion=False	
			permutacion.remove(valor_inicial)
			break		
	

	temp=(1-fi)*(feromona[valor_temp][valor_inicial])	
	actualizar_feromona=temp+fi*feromona_inicial
	feromona[valor_temp][valor_inicial]=float("%.5f"%(actualizar_feromona))
	feromona[valor_inicial][valor_temp]=float("%.5f"%(actualizar_feromona))
	return valor_inicial,condicion


def intensificacion(permutacion,permutacion_i,valor_inicial,condicion,final):
	valor_temp=valor_inicial
	argmax=[]
	size=len(permutacion)
	for i in range(size):
		t=feromona[valor_inicial][permutacion[i]]
		t=pow(t,alpha) 
		n=visibilidad[valor_inicial][permutacion[i]] 
		n=pow(n,beta) 
		argmax.append(t*n)

	mayor=np.argsort(argmax)
	valor_inicial=permutacion[mayor[size-1]]
	permutacion_i.append(valor_inicial)
	if valor_inicial==final:
		condicion=False
	permutacion.remove(valor_inicial)

	temp=(1-fi)*(feromona[valor_temp][valor_inicial])	
	actualizar_feromona=temp+fi*feromona_inicial
	feromona[valor_temp][valor_inicial]=float("%.5f"%(actualizar_feromona))
	feromona[valor_inicial][valor_temp]=float("%.5f"%(actualizar_feromona))
	return valor_inicial,condicion


def calcular_permutacion_i(permutacion,inicio,final):
	permutacion_i=[]
	valor_inicial=inicio
	permutacion_i.append(valor_inicial)
	permutacion.remove(valor_inicial)
	condicion=True
	while condicion and len(permutacion)!=0:
		q=np.random.uniform(0,1)
		if q>q0:
			valor_inicial,condicion=diversificacion(permutacion,permutacion_i,valor_inicial,condicion,final)	
		if q<=q0:
			valor_inicial,condicion=intensificacion(permutacion,permutacion_i,valor_inicial,condicion,final)
				
	return permutacion_i


def encontrar_arco(inicio, fin, permutacion):
	for r in range(len(permutacion)-1):
		if inicio==permutacion[r] or fin==permutacion[r]:
			if  inicio==permutacion[r+1] or fin==permutacion[r+1]:
				return True
							
	return False



def actualizar_feromona(permutacion_global, best_global):
	size=len(nodos)
	sumas=[0.0]
	
	for i in range(size-1):
		for j in range(i+1,size):
			inicio=nodos[i]
			fin=nodos[j]	
			suma_delta=p*feromona[inicio][fin]
			sumas.append(suma_delta)			
			
			if encontrar_arco(inicio,fin,permutacion_global): 	
				temp=(1-p)*(1.0/best_global)*100
				sumas.append(temp)
				
			else: 				
				sumas.append(0.0)	
				
			actualizar=sum(sumas)
			print(inicio,fin,"valor: ",actualizar)
			del sumas[:]
			feromona[inicio][fin]=float("%.5f"%(actualizar))
			feromona[fin][inicio]=float("%.5f"%(actualizar))



def generaciones():
	start_time2 = time.time()
	permutacion_hormigas=[0]*n_hormigas
	for i in range(n_hormigas):
		permutacion_hormigas[i]=[0]*len(nodos)

	fitness_hormigas=[0.0]*n_hormigas
	best_global=100000000	
	best_permutacion=[0]*len(nodos)
	
	for j in range(n_iteraciones):
		print("********************Iteracion: ",j+1,"**********************")
		for i in range(n_hormigas):
			permutacion=copy.copy(nodos)
			permutacion_hormigas[i]=calcular_permutacion_i(permutacion,inicio,final)
			fitness_hormigas[i]=fitness(permutacion_hormigas[i])
			print("Hormiga: ",i+1,permutacion_hormigas[i]," Costo: ", fitness_hormigas[i])

		indices=np.argsort(fitness_hormigas)
		if fitness_hormigas[indices[0]]<best_global:
			best_global=fitness_hormigas[indices[0]]
			best_permutacion=permutacion_hormigas[indices[0]]

		print("Mejor hormiga global:",best_permutacion,"costo:", best_global)
		if 'sca' in globals(): sca.remove()
		sca = plt.scatter(j, best_global, s=50, lw=0, c='blue', alpha=0.5); #plt.pause(0.001)
		actualizar_feromona(best_permutacion, best_global)
		
	print("Mejor hormiga global:",best_permutacion,"costo:", best_global)
	print time.time()-start_time2

#Parametros
p=0.9
q0=0.7
alpha = 0.1
beta = 0.1
fi=0.1
#q=1
feromona_inicial = 0.1
inicio=35
final=13
n_hormigas = 30
n_iteraciones = 30
visibilidad = m_visibilidad()
feromona = m_feromonas()
generaciones()
plt.xlabel('Iteraciones')
plt.ylabel('Funcion Aptitud')
plt.title('Algoritmo de la colonia de hormigas')
plt.ioff(); plt.show()


















