from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# - CARREGAR OS DADOS -
df = pd.read_csv(r'database\Cartao de credito.csv', encoding='ISO-8859-1', delimiter=";")
df_unlabeled = pd.read_csv(r'database\Dataset Validacao.csv', encoding='ISO-8859-1', delimiter=";")

# - TRATAR OS DADOS -
# Removendo a coluna de identificação da amostra
df.drop(columns=['Identificador da transação'], inplace=True)
df_unlabeled.drop(columns=['Identificador da transação'], inplace=True)

# Convertendo a coluna 'Bandeira' em variáveis numéricas usando codificação one-hot
df = pd.get_dummies(df, columns=['Bandeira do Cartão'], drop_first=True)
df_unlabeled = pd.get_dummies(df_unlabeled, columns=['Bandeira do Cartão'], drop_first=True)

# Trocando os valores SIM por 1 e NÃO por 0
df['Fraude'] = df['Fraude'].map({'SIM': 1, 'NÃO': 0})

# Convertendo as colunas numéricas para float
df['Distância de Casa'] = df['Distância de Casa'].str.replace(',', '.').astype(float)
df_unlabeled['Distância de Casa'] = df_unlabeled['Distância de Casa'].str.replace(',', '.').astype(float)
df['Distância da Última Transação'] = df['Distância da Última Transação'].str.replace(',', '.').astype(float)
df_unlabeled['Distância da Última Transação'] = df_unlabeled['Distância da Última Transação'].str.replace(',', '.').astype(float)
df['Razão entre o valor da compra e o valor médio'] = df['Razão entre o valor da compra e o valor médio'].str.replace(',', '.').astype(float)
df_unlabeled['Razão entre o valor da compra e o valor médio'] = df_unlabeled['Razão entre o valor da compra e o valor médio'].str.replace(',', '.').astype(float)

# Separando atributos e rótulos
X = df.drop(columns=['Fraude'])
X_unlabeled = df_unlabeled.drop(columns=['Fraude'])
y = df['Fraude']

# - SEPARAR OS DADOS EM TREINO E TESTE -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nNORMALIZANDO DADOS")

# - NORMALIZAR OS DADOS -
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_unlabeled = scaler.transform(X_unlabeled)

print("\nTREINANDO")

# - TREINAR O MODELO -
knn = KNeighborsClassifier(
    n_neighbors=5, 
    weights='distance', 
    algorithm='auto', 
    metric='euclidean')

knn.fit(X_train, y_train)

print("\nAVALIANDO")

# - FAZER PREVISÕES -
y_pred = knn.predict(X_test)

# Verificando acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAcurácia do modelo: {accuracy * 100:.2f}%')

# Verificando a matriz de confusão
class_names = ['NÃO', 'SIM']
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nMatriz de Confusão:')
print(pd.DataFrame(conf_matrix, columns=class_names, index=class_names))

# Verificando o relatório de classificação
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred, target_names=class_names))

# - CLASSIFICAR DADOS SEM RÓTULO -
print("CLASSIFICANDO DADOS SEM RÓTULO")
y_unlabeled_pred = knn.predict(X_unlabeled)

# Substituindo 0 por 'NÃO' e 1 por 'SIM'
y_pred_str = ['SIM' if x == 1 else 'NÃO' for x in y_unlabeled_pred]

# Adicionando a coluna de Fraude preenchida ao DataFrame
new_csv = pd.read_csv(r'database\Dataset Validacao.csv', encoding='ISO-8859-1', delimiter=';')
new_csv = new_csv.drop(columns=['Fraude'])
new_csv['Fraude'] = y_pred_str

# Salvando o resultado em um novo arquivo CSV
new_csv.to_csv(r'database\Dados classificados.csv', index=False, encoding='ISO-8859-1', sep=';')

print("\nCLASSIFICAÇÃO CONCLUÍDA\n")
