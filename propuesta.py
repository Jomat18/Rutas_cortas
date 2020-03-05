#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from numpy import loadtxt
import matplotlib.pyplot as plt
import time
#del a[i] i indice
# a.pop(i) i indice
# a.remove(elemento)

distancias = loadtxt('road_distance.txt')

nodos=[]
for i in range(226):  #226
	nodos.append(i)

n_nodos=len(nodos)

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
		actualizar_feromona(best_permutacion, best_global)

	return permutacion_hormigas

#Parametros
p=0.9
q0=0.7
alpha = 0.1
beta = 0.1
fi=0.1
q=1
feromona_inicial = 0.1
inicio=35
final=13
n_hormigas = 30
n_iteraciones = 2
visibilidad = m_visibilidad()
feromona = m_feromonas()


def generar_ruta(nodos_copy,inicio,final):
	if inicio==final:
		return -1

	temp=inicio
	ruta=[inicio]
	for i in range(n_nodos):
		nodo=np.random.choice(nodos_copy)
		nodos_copy.remove(nodo)
		if distancias[temp][nodo]!=0:
			ruta.append(nodo)
			temp=nodo

		if nodo==final:
			break

	return ruta


def ruta_valida(ruta,final):

	if ruta[len(ruta)-1]!=final:
		return True

	return False

def fitness(ruta):
	costo=0
	size=len(ruta)-1
	for i in range(size):
		costo+=distancias[ruta[i]][ruta[i+1]]

	return costo

def iniciar_poblacion(size,inicio,final):
	poblacion=[]
	print "********************* Poblacion Inicial **********************************"

	for i in range(size):
		condicion=True
		#nodos_=copy.copy(nodos)
		while condicion:
			nodos_=copy.copy(nodos)
			nodos_.remove(inicio)
			ruta=generar_ruta(nodos_,inicio,final)
 			condicion=ruta_valida(ruta,final)

		poblacion.append(ruta)
		print i+1, poblacion[i], "Costo: ", fitness(poblacion[i])

	print
	return poblacion


def seleccion_ruleta(poblacion,size):
	fitness_p=[]
	for i in range(size):
		fitness_p.append(fitness(poblacion[i]))
		print i+1, poblacion[i], "Costo: ", fitness_p[i]

	suma=0
	probabilidad=[0.0]*size
	padres=[]
	for i in range(size):
		probabilidad[i]=float("%.5f"%(1.0/fitness_p[i]))
		suma+=probabilidad[i]

	suma_probabilidad=0.0
	print "************************* Probabilidades *********************************"

	for i in range(size):
		probabilidad[i]=float("%.5f"%(((1.0)*probabilidad[i]/suma)))
		suma_probabilidad+=probabilidad[i]
		probabilidad[i]=suma_probabilidad
		print i+1,": Probabilidad: ", probabilidad[i]

	for n in range(size):
		aleatorio=np.random.uniform(0,1)
		for i in range(size):
			if aleatorio<probabilidad[i]:
				padres.append(poblacion[i])
				break

	return padres, fitness_p, poblacion


def pos(lista,c):
	tam=len(lista)
	for i in range(tam):
		if c==lista[i]:
			return i


def mesclar(pad1,pad2,corte):
	hijo1=[]
	hijo2=[]

	i=0
	while pad1[i]!=corte:
		hijo1.append(pad1[i])
		i+=1

	hasta=len(pad2)
	i=pos(pad2,corte)
	while hasta!=i:
		hijo1.append(pad2[i])
		i+=1

	i=0
	while pad2[i]!=corte:
		hijo2.append(pad2[i])
		i+=1

	hasta=len(pad1)
	i=pos(pad1,corte)
	while hasta!=i:
		hijo2.append(pad1[i])
		i+=1

	return hijo1, hijo2


def corte_valido(ruta, corte, condicion):
	for i in range(len(ruta)):
		if ruta[i]==corte:
			condicion=True
			return False, condicion

	return True, condicion


def generar_hijos(pad1,pad2):
	copia1=copy.copy(pad1)
	copia1.remove(inicio)
	copia1.remove(final)
	copia2=copy.copy(pad2)
	copia2.remove(inicio)
	copia2.remove(final)
	stop=True
	hacer_cruce=False

	while stop and len(copia1)!=0:
		corte=np.random.choice(copia1)
		stop,hacer_cruce=corte_valido(copia2,corte,hacer_cruce)
		#print "corte:", corte
		copia1.remove(corte)

	if hacer_cruce:
		pad1,pad2=mesclar(pad1,pad2,corte)
		#print "padre1", pad1
		#print "padre2", pad2

	return pad1, pad2

def cruzar(pos_c, padres):
	n_padres=len(pos_c)
	if n_padres%2==1:
		aleatorio=np.random.randint(len(padres))  #ojo <-len
		#print "agregado", aleatorio
		pos_c.append(aleatorio)

	n_padres=len(pos_c)/2
	print "************************ Cruzamiento ***************************"
	for i in range(n_padres):
		print "******************************* Padres ************************************"
		print 2*i+1, padres[pos_c[2*i]]
		print 2*i+2, padres[pos_c[2*i+1]]
		padres[pos_c[2*i]],padres[pos_c[2*i+1]]=generar_hijos(padres[pos_c[2*i]],padres[pos_c[2*i+1]])
		print "******************************** Hijos ************************************"
		print 2*i+1, padres[pos_c[2*i]]
		print 2*i+2, padres[pos_c[2*i+1]]


def cruzamiento(padres,size):
	#print "elegidos"
	posiciones_p=[]
	for i in range(size):
		aleatorio=np.random.uniform(0,1)
		if aleatorio<pc:
			#print i
			posiciones_p.append(i)

	cruzar(posiciones_p,padres)


def mutar(hijo,final):
	vertice=np.random.choice(hijo)
	#print "vertice", vertice
	hijo_mutado=[]
	pv=pos(hijo,vertice)
	condicion=True

	while condicion:
		nodos_=copy.copy(nodos)
		nodos_.remove(vertice)
		ruta=generar_ruta(nodos_,vertice,final)
		if ruta==-1:
			return hijo
 		condicion=ruta_valida(ruta,final)

	ip=0
	#print "ruta:",ruta
	while ip!=pv:
		hijo_mutado.append(hijo[ip])
		ip+=1

	hasta=len(ruta)
	for i in range(hasta):
		hijo_mutado.append(ruta[i])

	return hijo_mutado


def mutacion_opt(solucion, n_hijos, fit_solucion):
	temp=copy.copy(solucion)
	for i in range(n_hijos):
		temp=mutar(temp,final)
		valor=fitness(temp)
		if valor<fit_solucion:
			solucion=temp
			fit_solucion=valor
            		#break

	return solucion, fit_solucion


def escalada(solucion, n_escaladas, fit_solucion, n_hijos):
	for i in range(n_escaladas):
		solucion, fit_solucion=mutacion_opt(solucion,n_hijos,fit_solucion)

	return solucion, fit_solucion


def mutacion(hijos,size,n_escaladas,n_hijos):
	#print(hijos)
	for i in range(size):
		#aleatorio=np.random.uniform(0,1)
		#print hijos[i]
		#if aleatorio<pm:
		fit=fitness(hijos[i])
		hijos[i], fit_h = escalada(hijos[i], n_escaladas, fit, n_hijos)
		print "Mutacion:", hijos[i], "Costo:", fit_h


def evaluar(pob_inicial,hijos,fit_p,size):
	fit_h=[]
	for i in range(size):
		fit_h.append(fitness(hijos[i]))

	pob_result=[]
	costos1=list(np.argsort(fit_p))
	costos2=list(np.argsort(fit_h))
    	fit_actual=[]
	for i in range(size/2):
		pob_result.append(pob_inicial[costos1[i]])
        	fit_actual.append(fit_p[costos1[i]])
        	pob_result.append(hijos[costos2[i]])
        	fit_actual.append(fit_h[costos2[i]])

	return pob_result,fit_actual


def algoritmo_g(size,iteraciones,inicio,final,n_escalada,n_hijos):
	start_time2 = time.time()
	x=[]
	y=[]
	pob=generaciones()
	for i in range(iteraciones):
		print "*******************Iteracion: ",i+1,"***********************************"
		hijos,fit_p,pob_ini=seleccion_ruleta(pob,size)
		cruzamiento(hijos,size)
		mutacion(hijos,size,n_escalada,n_hijos)
		pob,fit=evaluar(pob_ini,hijos,fit_p,size)
		#pob=hijos
		ordenar=list(np.argsort(fit))
		
        	if 'sca' in globals(): sca.remove()
		x.append(i)
		y.append(fit[ordenar[0]])  #
		sca = plt.scatter(i, fit[ordenar[0]], s=50, lw=0, c='red', alpha=0.5); #plt.pause(0.001) 
		#,linewidth=2.5, linestyle="-", label="cosine"	
	print "***************************Poblacion Final**********************************"
	for i in range(size):
		print pob[i],"Costo: ",fitness(pob[i])
	print time.time()-start_time2


if __name__ == '__main__':
	pc=0.5
	pm=0.3
	size=n_hormigas
	iteraciones=28
	n_escalada=1
	n_hijos=5
	algoritmo_g(size,iteraciones,inicio,final,n_escalada,n_hijos)
	#plt.legend(loc='upper left')
	plt.xlabel('Iteraciones')
	plt.ylabel('Funcion Aptitud')
	plt.title('Algoritmo genetico hibrido')
	plt.ioff(); plt.show()

