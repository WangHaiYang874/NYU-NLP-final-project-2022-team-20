# Personality prediction: predicting the MBTI types over a reddit dataset

This is the final project of team 20, undergraduate nlp class in fall 2021, NYU. And we are doing personality prediction. This project was oringinally a [kaggle contest](https://www.kaggle.com/datasnaek/mbti-type), and then we read a [model with 73% accuracy](https://www.kaggle.com/zeyadkhalid/mbti-personality-types-classification-73-accuracy) that uses [type dynamics and cognitive functions](https://www.myersbriggs.org/my-mbti-personality-type/understanding-mbti-type-dynamics/type-dynamics.htm#:~:text=Type%20Dynamics.%20MBTI%C2%AE%20type%20is%20more%20than%20simply,an%20interrelated%20way%20to%20establish%20balance%20and%20effectiveness.) (I should find a better reference for type dynamics). Sort of inspired by the theory of type dynamics, we are going to use what we would like to call pseudo/generalized cognitive functions, that could be psychologically nonsense, to train a model. If the model works really well, then I guess that implies some of the pseudo coginitive functions are psychologically meaningful. 


### Team members

Students
- Oishika
- Haiyang
- Vincent
- Arthur

### links

- [google doc readme](https://docs.google.com/document/d/1UbfpTt0nYHkp2IjpMEiPJMHar7e8d3_kyFDOBw5yV8I/edit?usp=sharing) for more information
- [overleaf project proposal](https://www.overleaf.com/project/618d05ba58988c2754d187ec)
- [development set](https://github.com/WangHaiYang874/NYU-NLP-final-project-2022-team-20/blob/main/data/development.csv)
- [train set](https://drive.google.com/file/d/1SzXjA-yjqvkKfHglZyGnz2_PPt2M7vk0/view?usp=sharing)
- [test set](https://drive.google.com/file/d/1WYFT4TRwXrKvAQ7egR5RJdftB3QFbnzF/view?usp=sharing)

### TODOs

Please put your name next to the todo that you would like to do or you have done. Otherwise, the team won't give you credit for this project. 

- [x] the proposal: finished collectively
- [x] feature extraction/selection/data cleaning
    - [x] emoticons
    - [x] tfidf: (_Haiyang_ and _Vincent_)
    - [x] topic extraction: (_Vincent_)
    - [x] development dataset: (_Haiyang_)
    - [x] feature selection: (_Haiyang_)
    - [x] dimension reduction: done by _Vincent_. However, we decided that to not reduce the dimension. 
    - [x] parallel feature extracting (_Vincent_)
- [x] first layer models: (_Haiyang_, _Vincent_)
- [x] second layer model: this the the neural network taking input from previous layer to predict personality. (_Haiyang_)
    - [x] chunk max pool, 
    - [x] activations, 
    - [x] k max
    - [x] deeper models when the dataset is large enough?
    - [x] profiling the training process. 
    We discovered that the first layer model is just too good. So we decided that we are not using a NN for the second layer. Instead, we will use a simple random forest decision tree. 
    - [x] decision tree
- [x] building the model (_Vincent_, _Haiyang_)
    - [x] feature
    - [x] first layer
    - [x] second layer
- [x] evaluation (_Oishika_, _Vincent_)
- [x] presentation (_Oishika_, _Vincent_, _Haiyang_)
- [x] writing the paper (_Oishika_ 70%, others 30%)
