# Personality prediction: predicting the MBTI types over a reddit dataset

This is the final project of team 20, undergraduate nlp class in fall 2021, NYU. And we are doing personality prediction. This project was oringinally a [kaggle contest](https://www.kaggle.com/datasnaek/mbti-type), and then we read a [model with 73% accuracy](https://www.kaggle.com/zeyadkhalid/mbti-personality-types-classification-73-accuracy) that uses [type dynamics and cognitive functions](https://www.myersbriggs.org/my-mbti-personality-type/understanding-mbti-type-dynamics/type-dynamics.htm#:~:text=Type%20Dynamics.%20MBTI%C2%AE%20type%20is%20more%20than%20simply,an%20interrelated%20way%20to%20establish%20balance%20and%20effectiveness.) (I should find a better reference for type dynamics). Sort of inspired by the theory of type dynamics, we are going to use what we would like to call pseudo/generalized cognitive functions, that could be psychologically nonsense, to train a model. If the model works really well, then I guess that implies the some of the pseudo coginitive functions are psychologically meaningful. 


### Team members

Students
- Oishika
- Haiyang
- Vincent
- Arthur

### links

- [google doc readme](https://docs.google.com/document/d/1UbfpTt0nYHkp2IjpMEiPJMHar7e8d3_kyFDOBw5yV8I/edit?usp=sharing) for more information
- [overleaf project proposal](https://www.overleaf.com/project/618d05ba58988c2754d187ec)

### TODOs

Please put your name next to the todo that you would like to do or you have done. Otherwise, the team won't give you credit for this project. 

[x] the proposal: finished collectively
[ ] feature extraction/selection/data cleaning
    [ ] emoticons
    [ ] tfidf
    [x] development dataset: Done by Haiyang
    [ ] feature selection: Partly done by Haiyang
    [ ] what else?  
[ ] first layer models: these are the binary classifiers for pseudo-cognitive functions
[ ] second layer model: this the the neural network taking input from previous layer to predict personality
[ ] evaluation 
[ ] writing the paper
    [ ] draw graphs
[ ] reading some relevant psychological papers for ideas, perhaps? 