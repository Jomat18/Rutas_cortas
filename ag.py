#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from numpy import loadtxt
import matplotlib.pyplot as plt
#del a[i] i indice
# a.pop(i) i indice
# a.remove(elemento) 

distancias = loadtxt('road_distance.txt')

nodos=[]
for i in range(226):  #226
	nodos.append(i)

n_nodos=len(nodos)

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
	#temp=copy.copy(solucion)	
	#for i in range(n_hijos):
	solucion=mutar(solucion,final)
	fit_solucion=fitness(solucion)

	return solucion, fit_solucion


def escalada(solucion, n_escaladas, fit_solucion, n_hijos):
	for i in range(n_escaladas):
		solucion, fit_solucion=mutacion_opt(solucion,n_hijos,fit_solucion)

	return solucion, fit_solucion


def mutacion(hijos,size,n_escaladas,n_hijos):
	#print(hijos)
	for i in range(size):
		aleatorio=np.random.uniform(0,1)
		#print hijos[i]
		if aleatorio<pm:
			fit=fitness(hijos[i])
			hijos[i], fit_h = escalada(hijos[i], n_escaladas, fit, n_hijos)
			print "Mutacion:", hijos[i], "Costo:", fit_h


def evaluar(pob_inicial,hijos,fit_p,size):
	fit_h=[]
	for i in range(size):
		fit_h.append(fitness(hijos[i]))
		
	pob_result=[]
	costos1=np.argsort(fit_p)
	costos2=np.argsort(fit_h)
	fit_actual=[]
	for i in range(size/2):
		pob_result.append(pob_inicial[costos1[i]])
		fit_actual.append(fit_p[costos1[i]])
		pob_result.append(hijos[costos2[i]])
        	fit_actual.append(fit_h[costos2[i]])

	return pob_result,fit_actual
	

def algoritmo_g(size,iteraciones,inicio,final,n_escalada,n_hijos):
	pob=iniciar_poblacion(size,inicio,final)
	for i in range(iteraciones):	
		print "*******************Iteracion: ",i+1,"***********************************"
		hijos,fit_p,pob_ini=seleccion_ruleta(pob,size)
		cruzamiento(hijos,size)	
		mutacion(hijos,size,n_escalada,n_hijos)
		pob,fit=evaluar(pob_ini,hijos,fit_p,size)
		#pob=hijos
		ordenar=list(np.argsort(fit))
        	if 'sca' in globals(): sca.remove()
		sca = plt.scatter(i, fit[ordenar[0]], s=50, lw=0, c='red', alpha=0.5); #plt.pause(0.001) 
	print "***************************Poblacion Final**********************************"
	for i in range(size):
		print pob[i],"Costo: ",fitness(pob[i])	 	
	

if __name__ == '__main__':
	pc=0.5
	pm=0.3
	size=30
	iteraciones=30
	n_escalada=1
	inicio=100
	final=206
	n_hijos=1
	algoritmo_g(size,iteraciones,inicio,final,n_escalada,n_hijos)
	plt.ioff(); plt.show()
