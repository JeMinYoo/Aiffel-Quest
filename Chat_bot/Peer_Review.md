# AIFFEL Campus Code Peer Review Templete
- 코더 : 유제민, 김원영, 김주현
- 리뷰어 : 박진석, 김재이

    ※ 이 리뷰는 [첫 번째 파일](https://github.com/JeMinYoo/Aiffel-Quest/blob/master/Chat_bot/10_%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8%EB%A1%9C_%EB%A7%8C%EB%93%9C%EB%8A%94_%EB%8C%80%ED%99%94%ED%98%95_%EC%B1%97%EB%B4%87_%5B%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%5D.ipynb)을 주 대상으로 작성되었습니다. 다른 파일에서 첨부된 코드는 첨부구간에 출처를 명시했습니다.

# PRT(Peer Review Template)
[o]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
- 문제에서 요구하는 기능이 정상적으로 작동하는지?
    * 다음 조건들이 모두 충족되었다:
    1. 공백과 특수문자 처리, 토크나이징, 병렬데이터 구축의 과정이 적절히 진행되었다.
       1-1. 한국어에 맞게 데이터를 전처리하였다.
       1-2. SubwordTextEncoder 사용하여 토크나이징하였다.
    2. 구현한 트랜스포머 모델이 한국어 병렬 데이터 학습 시 안정적으로 수렴하였다.
    3. 한국어 입력문장에 맥락에 맞는 한국어로 답변을 리턴하였다.
    4. 대답을 얻는 예측 함수를 만들었다.
 
    * 코드 첨부:
  
    1-1.
  
      ```
        # 데이터 로드
        data = pd.read_csv('ChatbotData.csv')
        
        # 데이터 확인
        print(data.head())
        
        # 결측값 제거
        data = data.dropna()
        
        # 한글 외의 문자 제거
        def clean_text(text):
            text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', text)
            return text
        ...
      ```

    1-2.
  
      ```
      # TensorFlow Datasets SubwordTextEncoder를 사용하여 토크나이저 정의
      tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
          train_data['Q'].tolist() + train_data['A'].tolist(), target_vocab_size=2**13)
      ```

    2. 학습 시 안정적 수렴: 학습량이 적어 수치로는 평가하기 어려우나, 검증손실이 점차 감소해 1.1이하로 떨어지며 안정적으로 학습했음을 알 수 있다. 
      <img width="934" alt="스크린샷 2024-08-24 오후 9 24 23" src="https://github.com/user-attachments/assets/bca5d61c-140d-47c3-8df1-9748c2220661">
      
    3. 맥락에 맞는 답변: 맥락에 맞는 답을 얻는 데 성공했다. (존재하는 이슈가 있다면 답변의 반복과 항상 맥락에 맞지는 않는다는 점이 있겠다.)

      ```
        입력 : 사내커플인데 비밀연애임 답답해
        출력 : 헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다
        User: 사내커플인데 비밀연애임 답답해
        Chatbot: 헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다헤어진다

      ```
      
    5. 예측함수 decoder_inference() (원문 코드에는 주석과 함께 작성됨)

      ```
      def decoder_inference(sentence):
      sentence = preprocess_sentence(sentence)
      sentence = tf.expand_dims(
          START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
      output_sequence = tf.expand_dims(START_TOKEN, 0)
      for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, END_TOKEN[0]):
          break      
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)    
      return tf.squeeze(output_sequence, axis=0)
      ```
    

[o]  **2. 핵심적이거나 복잡하고 이해하기 어려운 부분에 작성된 설명을 보고 해당 코드가 잘 이해되었나요?**
- 해당 코드 블럭에 doc string/annotation/markdown이 달려 있는지 확인
- 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
- 주석을 보고 코드 이해가 잘 되었는지 확인

    * 모두 충족했으며 코드가 마크다운으로 세부적인 내용까지 잘 정리되어 있어 구조를 이해하기 쉽다. 예시:
    <img width="1146" alt="스크린샷 2024-08-24 오후 8 43 38" src="https://github.com/user-attachments/assets/2e70fa1b-17f9-4a41-a606-ed04a89a02c8">

    <img width="894" alt="스크린샷 2024-08-24 오후 9 00 14" src="https://github.com/user-attachments/assets/c82ee165-f5df-421d-96f0-53b07e52adbf">

        
[o]  **3. 에러가 난 부분을 디버깅하여 “문제를 해결한 기록”을 남겼나요? 또는 “새로운 시도 및 추가 실험”을 해봤나요?**
- 문제 원인 및 해결 과정을 잘 기록하였는지 확인
- 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인

    * 추가적으로 수행된 부분이 잘 작성되어 있다. 수행된 내용:
    1. KoNLPY 패키지의 Okt 모듈을 사용하여 한글 전처리
    2. CustomSchedule 개선
    3. 사전학습모델(GPT2) 파인튜닝
    4. 사전학습모델 로드 추가 시도 (코드는 없었음)

    * 코드:
    1. 추가 전처리: KoNLPY 패키지의 Okt 모듈을 사용하여 한글 전처리

      ```
      # 형태소 분석 및 불용어 제거
        okt = Okt()
        stopwords = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '한', '하다']
        
        def preprocess_text(text):
            tokens = okt.morphs(text)
            tokens = [word for word in tokens if word not in stopwords]
            return ' '.join(tokens)
        
        data['Q'] = data['Q'].apply(preprocess_text)
        data['A'] = data['A'].apply(preprocess_text)
      ```

    2. CustomSchedule 클래스에 get_config 모델을 추가

      ```
        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                ...
        
            def __call__(self, step):
                ...
        
            def get_config(self): # Add this method to make the schedule serializable
                return {
                    'd_model': self.d_model,
                    'warmup_steps': self.warmup_steps,
                }
      ```
  
    3. GPT2 모델을 파인튜닝한 시도 (PyTorch 사용) ([두 번째 파일](https://github.com/JeMinYoo/Aiffel-Quest/blob/master/Chat_bot/Chat_Bot.ipynb))
       
      ```
        # GPT-2 모델 로드
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        
        ...
        
        # Fine-tuning 과정
        epochs = 3
        for epoch in range(epochs):
            for batch in dataloader:
                inputs = batch.to(device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
      ```

      그 결과 예측을 생성한 바 있으며,
  
      ```
      챗봇:  안녕하면 좋은 아니까요.
      ```

      성능개선을 위한 다음을 포함한 시도가 기록되어 있다.
  
      ```
        # 입력과 출력 데이터를 하나의 텍스트로 결합, [BOS]와 [EOS] 토큰 사용
        df['input_output'] = "[BOS] " + df['Q'] + " [SEP] " + df['A'] + " [EOS]"
        
        # 토크나이저 로드
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'})
        
        # 텍스트를 토큰화
        df['tokens'] = df['input_output'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=128))
        
        class ChatbotDataset(Dataset):
            def __init__(self, tokens):
                self.tokens = tokens
        
            def __len__(self):
                return len(self.tokens)
        
            def __getitem__(self, idx):
                return torch.tensor(self.tokens[idx], dtype=torch.long)
      ```

      다음과 같은 결과를 얻었음을 확인할 수 있다.

      ```
        사용자: 오늘 날씨는 어때?
        챗봇:  오늘 날씨는 것도 좋
        
        사용자: 너의 이름은 뭐야?
        챗봇:  이름은 이름이 이름이 아
        
        사용자: 오늘 기분이 어때?
        챗봇:  오늘 아니에요. 
        
        사용자: 좋아하는 음식이 뭐야?
        챗봇:  좋아하는 음식이 �
        
        사용자: 인공지능이란 무엇인가요?
        챗봇:  인상은 인상�
        
        사용자: 어떤 영화를 추천해줄 수 있니?
        챗봇:  잘 있을 수 �
        
        사용자: 여행을 가고 싶은데 어디가 좋을까?
        챗봇:  사랑�
      ```

      이 이후에도 특수문자 제거, 공백 처리, 텍스트 추가적 토큰화, max_length 조정 등 추가적 시도가 이뤄지고 있음이 기록되어 있다.
      최종 결과는 다음과 같으며 약간 개선되었음을 확인 가능하다.

      ```
        사용자: 오늘 날씨는 어때?
        챗봇:  저는 거예요. 
        
        사용자: 너의 이름은 뭐야?
        챗봇:  이름이 마음이 마음이 있어요. 
        
        사용자: 오늘 기분이 어때?
        챗봇:  오늘를 연락이 아니라고 싶어요. 
        
        사용자: 좋아하는 음식이 뭐야?
        챗봇:  좋아하는 거예요. 
        
        사용자: 인공지능이란 무엇인가요?
        챗봇:  인공지능이란 무엇인가요. 
        
        사용자: 어떤 영화를 추천해줄 수 있니?
        챗봇:  그런 안 있는 거예요. 
        
        사용자: 여행을 가고 싶은데 어디가 좋을까?
        챗봇:  여행을 가고 싶은데 어디가 좋을까요. 

      ```
    

[o]  **4. 회고를 잘 작성했나요?**
- 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 상세히 기록 되어 있나요?
- 딥러닝 모델의 경우, 인풋이 들어가 최종적으로 아웃풋이 나오기까지의 전체 흐름을 도식화하여 모델 아키텍쳐에 대한 이해를 돕고 있는지 확인

    * 모두 충족했으며 [README.md](https://github.com/JeMinYoo/Aiffel-Quest/blob/master/Chat_bot/README.md) 페이지에 회고로 잘 드러나 있다.

[o]  **5. 코드가 간결하고 효율적인가요?**
- 파이썬 스타일 가이드 (PEP8)를 준수하였는지 확인
- 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 모듈화(함수화) 했는지
    - 잘 작성되었다고 생각되는 부분을 근거로 첨부합니다.
 
---

추가 코멘트:

박진석: 노드 프로젝트뿐만 아니라 다양한 모델, 케창딥을 활용등 여러가지 방법으로 시도를 했고 피어 리뷰하면서 구두로 설명해주셨습니다. 겪었던 오류들을 공유해주시면서 해결한 방법들을 같이 공유해주셔서 유익했습니다.

김재이: (팀 차원에서) 코랩 환경, LMS, 로컬에서 시도해봐 주셨습니다. 직접 구현한 트랜스포머 모델에서 한국어에 진행할 수 있는 다양한 개선 방법들을 적용해주셨고, 사전학습 모델을 불러온 케이스들에서는 내장 토크나이저를 사용하며 파인튜닝하고, 간단한 전처리를 통해 개선을 진행해주셨습니다. 또, 오류로서는 사전학습모델을 불러오는 경우 발생할 수 있는 버전 문제, 입력 차원의 문제에 대해서 공유해주셨습니다. 기본부터 응용까지 크게 또 자세히 훑어볼 수 있어 유익했습니다.
