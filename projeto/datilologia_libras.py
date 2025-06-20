import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import time

# ====== CONFIGURAÇÕES ======
LETRAS_PARA_COLETAR = list("WILLIAN")
FRAMES_POR_LETRA = 150  # aumentei para 150 amostras
DADOS_PATH = "dados_letras.csv"

# ====== MediaPipe ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def coletar_letra(letra):
    cap = cv2.VideoCapture(0)

    print(f"Prepare-se para mostrar a letra '{letra}' em 3 segundos...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    print(f"Mostre a letra '{letra}' agora. Capturando {FRAMES_POR_LETRA} amostras...")
    count = 0
    dados = []

    while count < FRAMES_POR_LETRA:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                dados.append(landmarks + [letra])
                count += 1

        cv2.putText(frame, f"Letra: {letra} | Coletadas: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Coleta de Dados", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
            break

    cap.release()
    cv2.destroyAllWindows()
    return dados

def coletar_dados():
    todos_dados = []
    for letra in LETRAS_PARA_COLETAR:
        dados = coletar_letra(letra)
        todos_dados.extend(dados)

    colunas = [f"x{i}" for i in range(len(todos_dados[0]) - 1)] + ["label"]
    df = pd.DataFrame(todos_dados, columns=colunas)
    df.to_csv(DADOS_PATH, index=False)
    print("Base de dados salva com sucesso.")

def treinar_modelo():
    df = pd.read_csv(DADOS_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print("Avaliação do modelo:\n", classification_report(y_test, y_pred))
    return modelo

def reconhecer_letras(modelo):
    cap = cv2.VideoCapture(0)
    texto_reconhecido = ""
    ultima_letra = ""
    ultimo_tempo = time.time()
    inicio = time.time()
    LIMIAR_CONFIANCA = 0.7  # só aceita predições com confiança maior que 0.7

    print("Mostre as letras com a mão. Pressione ESC para sair.")
    print("Aguardando 5 segundos para começar o reconhecimento...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                if len(landmarks) == modelo.n_features_in_:
                    probas = modelo.predict_proba([landmarks])[0]
                    indice = np.argmax(probas)
                    conf = probas[indice]
                    letra = modelo.classes_[indice]

                    tempo_atual = time.time()
                    if tempo_atual - inicio > 5 and conf > LIMIAR_CONFIANCA:
                        # permite repetir letra após 3 segundos
                        if (letra != ultima_letra) or (tempo_atual - ultimo_tempo > 3.0):
                            texto_reconhecido += letra
                            ultima_letra = letra
                            ultimo_tempo = tempo_atual

        cv2.putText(frame, f"Texto: {texto_reconhecido}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imshow("Reconhecimento em tempo real", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1 - Coletar dados")
    print("2 - Treinar modelo e reconhecer")
    opcao = input("Opção: ")

    if opcao == "1":
        coletar_dados()
    elif opcao == "2":
        if not os.path.exists(DADOS_PATH):
            print("Arquivo de dados não encontrado. Execute a coleta primeiro.")
        else:
            modelo = treinar_modelo()
            reconhecer_letras(modelo)
    else:
        print("Opção inválida.")
