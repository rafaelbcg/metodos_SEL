import numpy as np

'''
Solução desenvolvida por Rafael de Brito Cândido Gomes para aula de Solução Gauss
Seidel para sistema de equações lineares

e-mail: rafaeldebrito2@gmail.com

'''

def solucao_jacobi(matA,matB,precisao = 1e-6):
    '''
       matA é matriz de coeficientes;
       matB é matriz de termos independetes
       precisao é o nível de precisão que deseja obter a solução, por definição para precisão sendo 1e-6
    '''
    #preparando as varíaveis e matriz auxiliares
    n = len(matA)
    x = np.zeros((n),dtype=float)
    x2 = np.zeros((n),dtype=float)
    l = np.zeros((n, n),dtype=float)
    d = np.zeros((n, n),dtype=float)
    u = np.zeros((n, n),dtype=float)

    #preparar as matrizes iniciais
    for i in range(n):
        d[i, i] = 1 / matA[i, i]
        for j in range(n):
            if i > j:
                l[i, j] = matA[i, j]*d[i,i]
            elif i < j:
                u[i, j] = matA[i, j]*d[i,i]

    print('Calculando a solução: ')
    for count in range(100):
        print(f'{count+1} -- {x}')

        # soma das matrizes
        x_next = -l*x-u*x+d*matB
        for i in range(n):
            soma = 0
            for j in range(n):
                soma += x_next[i,j]
            x2[i] = soma

        #verificando a questão convergência com a precisão    
        maior = np.amax(x2)
        dif = abs(x-x2)
        maior_dif = np.amax(dif)

        if maior_dif/maior < precisao:
            return x2
        x = x2

    print('Não foi possível encontrar a solução usando Gauss-Seidel')
    return None

#----------- codigo inicial
matA = np.array([[10, 1, -1], [1, 15, 1], [-1, 1, 20]],dtype=float)
matB = np.array([18,-12,17], dtype=float)
solucao = solucao_jacobi(matA,matB,0.01)

print(f"A solução do sistema {matA} * x = {matB} é {solucao.round(4)}")

