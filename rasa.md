# rasa

### 安装配置(Python3)

安装思路：使用python的virtualenv来作为环境，目的是避免不同的应用需要不同的python包版本问题，但在运行

```
python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
```

的时候出现了问题

```
Exception: Not all required packages are installed. To use this pipeline, you need to install the missing dependencies. Please install sklearn_crfsuite
```

网上查资料显示用anaconda，可以解决，并且也可以达到同样的目的，所以使用anaconda安装。

anaconda直接在官网上的步骤安装，再安装完conda后，创建一个rasa的环境，然后进入

```
liuxinyideMacBook-Pro:~ liuxinyi$ source activate rasa
```

```
(rasa) liuxinyideMacBook-Pro:~ liuxinyi$ pip install rasa_core
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ pip install rasa_nlu[tensorflow]
```

在anaconda3/envs/rasa目录中，把

```
## story_happy
* greet
  - utter_greet
* mood_happy
  - utter_happy

## story_unhappy
* greet2
  - utter_greet
*  mood_unhappy
  - utter_unhappy
```

存入stories.md中

把

```
intents:
  - greet
  - mood_happy
  - mood_unhappy

actions:
- utter_greet
- utter_happy
- utter_unhappy

templates:
  utter_greet:
  - text: "你好，你今天过的怎么样"

  utter_happy:
  - text: "那很棒棒哦"

  utter_unhappy:
  - text: "咋了，可以告诉我吗"
```

存入domain.yml中

然后训练Core模型，训练的模型会存在该目录models/dialogue下

```
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue
```

这个时候已经可以识别固定的意图比如，前面定义的greet

```
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ python -m rasa_core.run -d models/dialogue
```

```
2019-01-02 15:33:22.498060: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-02 15:33:24 INFO     root  - Rasa Core server is up and running on http://localhost:5005
Bot loaded. Type a message and press enter (use '/stop' to exit): 
/greet
你好，你今天过的怎么样
127.0.0.1 - - [2019-01-02 15:33:35] "POST /webhooks/rest/webhook?stream=true&token= HTTP/1.1" 200 234 0.158596
/mood_happy
那很棒棒哦
127.0.0.1 - - [2019-01-02 15:33:40] "POST /webhooks/rest/webhook?stream=true&token= HTTP/1.1" 200 198 0.011773
```

接下来添加nlu模块

把

```
## intent:greet
- 你好
- 上午好
- 下午好
- 早上好
- 晚上好

## intent:mood_happy
- 很好
- 我很好

## intent:mood_unhappy
- 我很难受
- 我心情很差
```

存到该目录的nlu.md中

除此之外，我们还需要一个nlu的配置文件，`nlu_config.yml`，由于我们是中文系统，所以language对应的是zh，如果你需要英文的对话请修改为en。

```
language: zh
pipeline: tensorflow_embedding
```

存入nlu_config.yml中，运行下面的指令进行训练

```
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
```

但此时还是出现了同样的错误

```
Exception: Not all required packages are installed. To use this pipeline, you need to install the missing dependencies. Please install sklearn_crfsuite
```

这时候

```
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ conda install sklearn_crfsuite
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ conda install -c conda-forge pysoundfile
```

然后再次运行

```
(rasa) liuxinyideMacBook-Pro:rasa liuxinyi$ python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
```

然后运行

```
python -m rasa_core.run -d models/dialogue -u models/current/nlu
```

成功进入交互界面

```
2019-01-02 15:47:31 INFO     root  - Rasa Core server is up and running on http://localhost:5005
Bot loaded. Type a message and press enter (use '/stop' to exit): 
你好
你好，你今天过的怎么样
127.0.0.1 - - [2019-01-02 15:47:38] "POST /webhooks/rest/webhook?stream=true&token= HTTP/1.1" 200 234 0.204469
很好
那很棒棒哦
127.0.0.1 - - [2019-01-02 15:47:44] "POST /webhooks/rest/webhook?stream=true&token= HTTP/1.1" 200 198 0.009246
下午好
你好，你今天过的怎么样
127.0.0.1 - - [2019-01-02 15:47:48] "POST /webhooks/rest/webhook?stream=true&token= HTTP/1.1" 200 234 0.008973
```

#### example

1. You can train the Rasa NLU model by running:  
   ```make train-nlu```  
   This will train the Rasa NLU model and store it inside the `/models/current/nlu` folder of your project directory.

2. Train the Rasa Core model by running:  
   ```make train-core```  
   This will train the Rasa Core model and store it inside the `/models/current/dialogue` folder of your project directory.

3. In a new terminal start the server for the custom action by running:  
   ```make action-server```  
   This will start the server for emulating the custom action.

4. Test the assistant by running:  
   ```make cmdline```  
   This will load the assistant in your terminal for you to chat.


## Rasa NLU

### 选择一个nlu pipeline

#### pipeline的解释

用来训练词向量，有两种，一种是spacy_sklearn，用来训练少量数据，没有太多训练数据可以用这个。

另一种tensorflow_embedding，可以训练专业的特有的大量的数据（定制）

```
language: "en"
pipeline: "spacy_sklearn"
```

```
language: "en"
pipeline: "tensorflow_embedding"
```

#### 多重意图

如果想分割意图成多种标签，只能使用tensorflow_embedding

```
language: "en"

pipeline:
- name: "intent_featurizer_count_vectors"
- name: "intent_classifier_tensorflow_embedding"
  intent_tokenization_flag: true
  intent_split_symbol: "+"
```

intent_tokenization_flag: true      将意图标签拆分为标记

intent_split_symbol: "+"                 设置分隔符拆分意图标签

#### 预先定义的pipeline（意图分类器）

```
language: "en"

pipeline: "spacy_sklearn"
```

这种是一种模版和下面这种是等价的

```
language: "en"

pipeline:
- name: "nlp_spacy"
- name: "tokenizer_spacy"
- name: "intent_entity_featurizer_regex"
- name: "intent_featurizer_spacy"
- name: "ner_crf"
- name: "ner_synonyms"
- name: "intent_classifier_sklearn"
```

##### spacy_sklearn

有模版，对语言有要求

##### tensorflow_embedding

有模版，并且支持可以标记化的任何语言

```
language: "en"

pipeline:
- name: "tokenizer_whitespace"
- name: "ner_crf"
- name: "ner_synonyms"
- name: "intent_featurizer_count_vectors"
- name: "intent_classifier_tensorflow_embedding"
```

如果有标记，可以替换tokenizer_whitespace

##### mitie &mitie_sklearn

无模版，需要自己写配置

```
language: "en"

pipeline:
- name: "nlp_mitie"
  model: "data/total_word_feature_extractor.dat"
- name: "tokenizer_mitie"
- name: "ner_mitie"
- name: "ner_synonyms"
- name: "intent_entity_featurizer_regex"
- name: "intent_classifier_mitie"
```

```
language: "en"

pipeline:
- name: "nlp_mitie"
  model: "data/total_word_feature_extractor.dat"
- name: "tokenizer_mitie"
- name: "ner_mitie"
- name: "ner_synonyms"
- name: "intent_entity_featurizer_regex"
- name: "intent_featurizer_mitie"
- name: "intent_classifier_sklearn"
```

##### Custom pipelines

做实体辨别，不会做意图分类，而且是直接列出用的

```
pipeline:
- name: "nlp_spacy"
- name: "ner_crf"
- name: "ner_synonyms"
```

### 语言支持

在我们自己的领域训练，所以任何语言都可以（tensorflow_embedding）

使用tensorflow_embedding有两种方式

一种是模版

```
language: "en"

pipeline: "tensorflow_embedding"
```

一种是定义配置文件，把能用到的组件都列下来

```
language: "en"

pipeline:
- name: "tokenizer_whitespace"
- name: "ner_crf"
- name: "ner_synonyms"
- name: "intent_featurizer_count_vectors"
- name: "intent_classifier_tensorflow_embedding"
```

#### 预先训练好的词向量—中文的使用

spacy

spacy-sklearn 对应支持的语言有english (`en`), german (`de`), spanish (`es`), portuguese (`pt`), italian (`it`), dutch (`nl`), french (`fr`)

MITIE对应 english (`en`)

jieba-MITIE 对应chinese (`zh`)

#### 添加一种新的语言

https://rasa.com/docs/nlu/languages/

### 实体抽取

| Component           | Requires         | Model                    | notes                             |
| ------------------- | ---------------- | ------------------------ | --------------------------------- |
| `ner_crf`           | sklearn-crfsuite | conditional random field | good for training custom entities |
| `ner_spacy`         | spaCy            | averaged perceptron      | provides pre-trained entities     |
| `ner_duckling_http` | running duckling | context-free grammar     | provides pre-trained entities     |
| `ner_mitie`         | MITIE            | structured SVM           | good for training custom entities |

#### custom entities（客户实体||自定义实体）

同一个词在不同环境下意思不同

ner_crf 组件可以在任何语言中学习custom entities

#### 提取地点，数据，人物，组织

spacy可以在不同的语言中预训练命名实体识别器，但是不建议训练自己的NER

#### 日期，金额，持续时间，距离，ordinals

duckling可以把像”next Thursday at 8pm“转化成

```
"next Thursday at 8pm"
=> {"value":"2018-05-31T20:00:00.000+01:00"}
```

#### 正则表达式

在训练数据的时候可以提供正则表达式，让模型更专注这些信息，可以更快的建立连接

但是如果只是想精准的匹配，则只用在后续的代码加就行，不用把它作为一个训练的参数

#### 返回的实体对象

```
{
  "text": "show me chinese restaurants",
  "intent": "restaurant_search",
  "entities": [
    {
      "start": 8,
      "end": 15,
      "value": "chinese",
      "entity": "cuisine",
      "extractor": "ner_crf",
      "confidence": 0.854,
      "processors": []
    }
  ]
}
```

这两个字段显示有关管道如何影响返回的实体的信息

extractor The `extractor` field of an entity tells you which entity extractor found this particular entity

哪个实体提取器发现的

Processors The `processors` field contains the name of components that altered this specific entity.

更改具体实体的组件

其他的提取器，像duckling，也许会有更多的信息

```
{
  "additional_info":{
    "grain":"day",
    "type":"value",
    "value":"2018-06-21T00:00:00.000-07:00",
    "values":[
      {
        "grain":"day",
        "type":"value",
        "value":"2018-06-21T00:00:00.000-07:00"
      }
    ]
  },
  "confidence":1.0,
  "end":5,
  "entity":"time",
  "extractor":"ner_duckling_http",
  "start":0,
  "text":"today",
  "value":"2018-06-21T00:00:00.000-07:00"
}
```

### 评估和提升模型

#### 从反馈中提升模型

一旦有一个版本的机器人在运行，rasa nlu服务器会把每一条请求记录存在目录parse下的文件里，默认情况下，这些文件保存在文件夹日志中。

```
{
  "user_input":{
    "entities":[]   ],
    "intent":{
      "confidence":0.32584617693743012,
      "name":"restaurant_search"
    },
    "text":"nice thai places",
    "intent_ranking":[ ... ]
  },
  ...
  "model":"default",
  "log_time":1504092543.036279
}
```

用户所说的内容是改进模型的最佳训练数据来源，必须手动完成每个预测并在把这些数据添加到训练集之前纠正，上面的例子是说，thai没有被当作一个菜（应该是泰国菜）。

#### 评估模型

rasa nlu有一个评估模块，机器学习中的标准技术是将一些数据作为测试集分开，可以用下面这个命令查看模型预测测试用例的情况

```
python -m rasa_nlu.evaluate \
    --data data/examples/rasa/demo-rasa.json \
    --model projects/default/model_20180323-145833
```

--data 是指训练数据

--model 是指训练模型

假如没有独立的测试集，可以使用交叉验证来估计模型的优化程度。运行--mode crossvalidation这个评估脚本。（交叉验证标志）

```
python -m rasa_nlu.evaluate \
    --data data/examples/rasa/demo-rasa.json \
    --config sample_configs/config_spacy.yml \
    --mode crossvalidation
```

 但是在这个模式下不能指定模型model，因为将针对每个交叉验证折叠对部分数据进行新模型培训（because a new model will be trained on part of the data for every cross-validation fold.）。

#### 意图分类

### 杀死端口号

```
ps -fA | grep python

kill 端口号
```