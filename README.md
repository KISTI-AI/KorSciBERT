
# 과학기술분야 BERT 사전학습 모델 (KorSci BERT)
 본 KorSci BERT 언어모델은 한국과학기술정보연구원과 한국특허정보원이 공동으로 연구한 과제의  결과물 중 하나로, 기존 [Google BERT base](https://github.com/google-research/bert) 모델의 아키텍쳐를 기반으로, 한국 논문 & 특허 코퍼스 총 97G (약 3억 8천만 문장)를 사전학습한 모델이다.

## Train dataset
|Type|Corpus|Sentences|Avg sent length|
|--|--|--|--|
|논문|15G|72,735,757|122.11|
|특허|82G|316,239,927|120.91|
|합계|97G|388,975,684|121.13|

## Model architecture
-   attention_probs_dropout_prob:0.1
-   directionality:"bidi"
-   hidden_act:"gelu"
-   hidden_dropout_prob:0.1
-   hidden_size:768
-   initializer_range:0.02
-   intermediate_size:3072
-   max_position_embeddings:512
-   num_attention_heads:12
-   num_hidden_layers:12
-   pooler_fc_size:768
-   pooler_num_attention_heads:12
-   pooler_num_fc_layers:3
-   pooler_size_per_head:128
-   pooler_type:"first_token_transform"
-   type_vocab_size:2
-   vocab_size:15330

## Vocabulary
 - Total 15,330 words
 - Included special tokens ( [PAD], [UNK], [CLS], [SEP], [MASK] )
 - File name : vocab_kisti.txt

## Language model
- Model file : model.ckpt-262500 (Tensorflow ckpt file)

## Pre training
- Trained 128 Seq length 1,600,000 + 512 Seq length 500,000 스텝 학습
- 논문+특허 (97 GB) 말뭉치의 3억 8천만 문장 데이터 학습
- NVIDIA V100 32G 8EA GPU 분산학습 with [Horovod Lib](https://github.com/horovod/horovod)
- NVIDIA [Automixed Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision) 방식 사용

## Downstream task evaluation
본 언어모델의 성능평가는 과학기술표준분류 및 특허 선진특허분류([CPC](https://www.kipo.go.kr/kpo/HtmlApp?c=4021&catmenu=m06_07_01)) 2가지의 태스크를 파인튜닝하여 평가하는 방식을 사용하였으며, 그 결과는 아래와 같다.
|Type|Classes|Train|Test|Metric|Train result|Test result|
|--|--|--|--|--|--|--|
|과학기술표준분류|86|130,515|14,502|Accuracy|68.21|70.31|
|특허CPC분류|144|390,540|16,315|Accuracy|86.87|76.25|


# 과학기술분야 토크나이저 (KorSci Tokenizer)

본 토크나이저는 한국과학기술정보연구원과 한국특허정보원이 공동으로 연구한 과제의  결과물 중 하나이다.  그리고, 위 사전학습 모델에서 사용된 코퍼스를 기반으로 명사 및 복합명사 약 600만개의 사용자사전이 추가된 [Mecab-ko Tokenizer](https://bitbucket.org/eunjeon/mecab-ko/src/master/)와 기존 [BERT WordPiece Tokenizer](https://github.com/google-research/bert)가 병합되어진 토크나이저이다.

##  모델 다운로드
http://doi.org/10.23057/46

##  요구사항

### 은전한닢 Mecab 설치 & 사용자사전 추가
	Installation URL: https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/
	mecab-ko > 0.996-ko-0.9.2
	mecab-ko-dic > 2.1.1
	mecab-python > 0.996-ko-0.9.2

### 논문 & 특허 사용자 사전
- 논문 사용자 사전 : pap_all_mecab_dic.csv (1,001,328 words)
- 특허 사용자 사전 : pat_all_mecab_dic.csv (5,000,000 words)

### konlpy  설치
	pip install konlpy
	konlpy > 0.5.2

##  사용방법
	import tokenization_kisti as tokenization
	 
	vocab_file = "./vocab_kisti.txt"  

	tokenizer = tokenization.FullTokenizer(  
							vocab_file=vocab_file,  
							do_lower_case=False,  
							tokenizer_type="Mecab"  
						)  
  
	example = "본 고안은 주로 일회용 합성세제액을 집어넣어 밀봉하는 세제액포의 내부를 원호상으로 열중착하되 세제액이 배출되는 절단부 쪽으로 내벽을 협소하게 형성하여서 내부에 들어있는 세제액을 잘짜질 수 있도록 하는 합성세제 액포에 관한 것이다."  
	tokens = tokenizer.tokenize(example)  
	encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)  
	decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_tokens)  
	  
	print("Input example ===>", example)  
	print("Tokenized example ===>", tokens)  
	print("Converted example to IDs ===>", encoded_tokens)  
	print("Converted IDs to example ===>", decoded_tokens)
	
	============ Result ================
	Input example ===> 본 고안은 주로 일회용 합성세제액을 집어넣어 밀봉하는 세제액포의 내부를 원호상으로 열중착하되 세제액이 배출되는 절단부 쪽으로 내벽을 협소하게 형성하여서 내부에 들어있는 세제액을 잘짜질 수 있도록 하는 합성세제 액포에 관한 것이다.
	Tokenized example ===> ['본', '고안', '은', '주로', '일회용', '합성', '##세', '##제', '##액', '을', '집', '##어', '##넣', '어', '밀봉', '하', '는', '세제', '##액', '##포', '의', '내부', '를', '원호', '상', '으로', '열', '##중', '착', '##하', '되', '세제', '##액', '이', '배출', '되', '는', '절단부', '쪽', '으로', '내벽', '을', '협', '##소', '하', '게', '형성', '하', '여서', '내부', '에', '들', '어', '있', '는', '세제', '##액', '을', '잘', '짜', '질', '수', '있', '도록', '하', '는', '합성', '##세', '##제', '액', '##포', '에', '관한', '것', '이', '다', '.']
	Converted example to IDs ===> [59, 619, 30, 2336, 8268, 819, 14100, 13986, 14198, 15, 732, 13994, 14615, 39, 1964, 12, 11, 6174, 14198, 14061, 9, 366, 16, 7267, 18, 32, 307, 14072, 891, 13967, 27, 6174, 14198, 14, 698, 27, 11, 12920, 1972, 32, 4482, 15, 2228, 14053, 12, 65, 117, 12, 4477, 366, 10, 56, 39, 26, 11, 6174, 14198, 15, 1637, 13709, 398, 25, 26, 140, 12, 11, 819, 14100, 13986, 377, 14061, 10, 487, 55, 14, 17, 13]
	Converted IDs to example ===> ['본', '고안', '은', '주로', '일회용', '합성', '##세', '##제', '##액', '을', '집', '##어', '##넣', '어', '밀봉', '하', '는', '세제', '##액', '##포', '의', '내부', '를', '원호', '상', '으로', '열', '##중', '착', '##하', '되', '세제', '##액', '이', '배출', '되', '는', '절단부', '쪽', '으로', '내벽', '을', '협', '##소', '하', '게', '형성', '하', '여서', '내부', '에', '들', '어', '있', '는', '세제', '##액', '을', '잘', '짜', '질', '수', '있', '도록', '하', '는', '합성', '##세', '##제', '액', '##포', '에', '관한', '것', '이', '다', '.']
	
	
### Fine-tuning with KorSci-Bert
- [Google Bert](https://github.com/google-research/bert)의 Fine-tuning 방법 참고
- Sentence (and sentence-pair) classification tasks: "run_classifier.py" 코드 활용
- MRC(Machine Reading Comprehension) tasks: "run_squad.py" 코드 활용
