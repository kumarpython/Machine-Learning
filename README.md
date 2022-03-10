# Machine Learning | Machine Learning with Python + Numpy(Vectors) | Machine Learning with Python + Scikitlearn

## Machine Learning with Python + Vectors

If you work in machine learning, you will need to work with vectors. There’s almost no ML model where vectors aren’t used at some point in the project lifecycle. 

And while vectors are used in many other fields, there’s something different about how they’re used in ML. This can be confusing. The potential confusion with vectors, from an ML perspective, is that we use them for different reasons at different stages of an ML project. 

This means that a strictly mathematical definition of vectors can fail to convey all the information you need to work with and understand vectors in an ML context. For example, if we think of a simple lifecycle of a typical ML project.

there are three different stages. Each one has a slightly different use case for vectors. In this article, we’ll clear this all up by looking at vectors in relation to these stages:

### Input:
Machines can’t read text or look at images like you and me. They need input to be transformed or encoded into numbers. Vectors, and matrices (we’ll get to these in a minute) represent inputs like text and images as numbers, so that we can train and deploy our models. We don’t need a deep mathematical understanding of vectors to use them as a way to encode information for our inputs. We just need to know how vectors relate to features, and how we can represent those features as a vector. 
### Model: 
The goal of most ML projects is to create a model that performs some function. It could be classifying text, or predicting house prices, or identifying sentiment. In deep learning models, this is achieved via a neural network where the neural network layers use linear algebra (like matrix and vector multiplication) to tune your parameters. This is where the mathematical definition of vectors is relevant for ML. We won’t get into the specifics of linear algebra in this post, but we’ll look at the important aspects of vectors and matrices which we need to work with these models. This includes understanding vector spaces and why they’re important for ML. 
### Output: 
The output of our ML model can be a range of different entities depending on our goal. If we’re predicting house prices, the output will be a number. If we’re classifying images, the output will be a category of image. The output, however, can be a vector as well. For example, NLP models like the Universal Sentence Encoder (USE) accept text and then output a vector (called an embedding) representing the sentence. You can then use this vector to perform a range of operations, or as an input into another model. Among the operations you can perform are clustering similar sentences together in a vector space, or finding similarity between different sentences using operations like cosine similarity. Understanding these operations will help you know how to work with models that output vectors like these NLP models.
Note: you can find all the code for this post on Github.

### Scalars, vectors and matrices
Vectors are not the only way to represent numbers for machines to process and transform inputs. While we’re mainly concerned with vectors in this post, we’ll need to define other structures that represent numbers also.

### Scalars: 
For our purposes, scalars are just numbers. We can think of them like any regular value we use. Any single value from our dataset would represent a scalar. The number of bedrooms in our house price data, for example, would be a scalar. If we only used one feature as an input from our house price data, then we could represent that as a scalar value.
This will help us in the following sections, where we need to understand how vectors interact with those other structures. For the purposes of this section, let’s take the Kaggle dataset on house prices as our frame of reference, and see how we would represent this data using scalars, vectors and matrices.

### Vectors: 
There seems to be more than one usable feature in our house price data. How would we represent multiple features? The total square footage of the house would be a useful piece of information to have when trying to predict a house price. In its most simple format, we can think of a vector as a 1-D data structure. We’ll define a vector in more detail shortly

### Matrices: 
So far we’ve just looked at the first house in our dataset. What if we need to pass in batches of multiple houses, with their bedroom and square foot / meter values? This is where matrices come in. You can think of a matrix as a 2-D data structure, where the two dimensions refer to the number of rows and columns

### Inputs: Vectors as encoders
As we noted earlier, the way we use vectors in ML is very dependent on whether we are dealing with inputs, outputs, or the model itself. When we use vectors as inputs, the main use is their ability to encode information in a format that our model can process, and then output something useful to our end goal. Let’s look at how vectors are used as encoders with a simple example.

Imagine we want to create an ML model which writes new David Bowie songs. To do this, we would need to train the model on actual David Bowie songs, and then prompt it with an input text which the model will then “translate” into a David Bowie-esque lyric. 

### How do we create our input vectors?
We need to pass a load of David Bowie lyrics to our model so that it can learn to write like Ziggy Stardust. We know that we need to encode the information in the lyrics into a vector, so that the model can process it, and the neural network can start doing lots of math operations on our inputs to tune parameters. Another way to think of this is that we’re using the vector to represent the features we want the model to learn. 

The model could be trying to predict house prices, using the house price dataset we used earlier to explain the difference between scalars, vectors and matrices. In that case, we might pass it information such as the house size, number of bedrooms, postcode and things like this. Each of these is a feature that might help the model predict a more accurate house price. It’s easy to see how we would create an input vector for our house price model. 

### Model: Vectors as transformers
At this point we’ve represented our input as a vector, and want to use it to train our model. In other words we want our model to learn a transformation which uses the features in our input to return an output that achieves some goal. We’ve already discussed an example of such goals:

Predict house prices: We showed how we could create a vector which encoded the features needed from our house price data. In this scenario, we wanted to learn a transformation that used those features to output a predicted price for that house.
Create Bowie-esque lyrics: For our Bowie model, output was text, and the transformation was to turn an input sentence into a Bowie-esque lyric.
Sentence encoder: The USE model transforms an input sentence into an output vector. Its goal is to create an output sentence which, using the features of the input sentence, represents that sentence in something called a vector space. 
Neural networks are a way to create models which can learn these transformations, so we can turn our house price data into predictions, and our dull sentences into Bowie-like musings. 

Specifically, it’s the hidden layers of neural networks which take in our vectorized inputs, create a weight matrix, and then use that weight matrix to create our desired outputs:

Hidden layers
The hidden layer is where the computation takes place to learn the transformations we need to create our outputs. Source DeepAi.org
We don’t need to get into the specifics of how neural networks work. But we do need to learn more about vectors and matrices, and how they interact, so we can understand why we use them and – more importantly – how we can use the output of these networks. As a result, in this section we’re going to:

### Define vectors: 
We need to briefly define vectors from a mathematical perspective, and show how the context is different from how we used them as inputs. 
Vector spaces: Vectors live in vector spaces. If you understand how these spaces work, then you’ll be able to understand most of the reasons why vectors are important in ML.
Vector and matrix operations: We manipulate vectors in these spaces by doing things like multiplying them with matrices. They squeeze, squash, move and transform vectors until our models have learned something we can use for our outputs. These operations will be important for the final section, where we will look at common operations which we use when our models output vectors, like the USE model did with our Bowie lyrics.
The goal of this section is to help us understand that the way we use vectors to learn functions in our ML models is slightly different from the way we use vectors in our inputs and our outputs. 
This is why the context of the pipeline stage in your ML project is important. The useful knowledge about vectors can be different depending on what you want to do with your vector. Knowing that will, hopefully, help you with your future ML projects.
### What is a vector?
Previously we defined a vector as a list of numbers or a 1-D data structure. This helped us understand how we could encode information from our datasets as inputs to pass to our ML models. 
Within these models, once that input is received, it’s important to understand that vectors are objects which have the unusual distinction of having both a magnitude and a direction.

### Model: Vectors as transformers
At this point we’ve represented our input as a vector, and want to use it to train our model. In other words we want our model to learn a transformation which uses the features in our input to return an output that achieves some goal. We’ve already discussed an example of such goals:
Predict house prices: We showed how we could create a vector which encoded the features needed from our house price data. In this scenario, we wanted to learn a transformation that used those features to output a predicted price for that house.
Create Bowie-esque lyrics: For our Bowie model, output was text, and the transformation was to turn an input sentence into a Bowie-esque lyric.
Sentence encoder: The USE model transforms an input sentence into an output vector. Its goal is to create an output sentence which, using the features of the input sentence, represents that sentence in something called a vector space. 
Neural networks are a way to create models which can learn these transformations, so we can turn our house price data into predictions, and our dull sentences into Bowie-like usings. 
Specifically, it’s the hidden layers of neural networks which take in our vectorized inputs, create a weight matrix, and then use that weight matrix to create our desired outputs:
### Hidden layers
The hidden layer is where the computation takes place to learn the transformations we need to create our outputs. Source DeepAi.org
We don’t need to get into the specifics of how neural networks work. But we do need to learn more about vectors and matrices, and how they interact, so we can understand why we use them and – more importantly – how we can use the output of these networks. As a result, in this section we’re going to:

### Define vectors: 
We need to briefly define vectors from a mathematical perspective, and show how the context is different from how we used them as inputs. 
Vector spaces: Vectors live in vector spaces. If you understand how these spaces work, then you’ll be able to understand most of the reasons why vectors are important in ML.
Vector and matrix operations: We manipulate vectors in these spaces by doing things like multiplying them with matrices. They squeeze, squash, move and transform vectors until our models have learned something we can use for our outputs. These operations will be important for the final section, where we will look at common operations which we use when our models output vectors, like the USE model did with our Bowie lyrics.
The goal of this section is to help us understand that the way we use vectors to learn functions in our ML models is slightly different from the way we use vectors in our inputs and our outputs. 
This is why the context of the pipeline stage in your ML project is important. The useful knowledge about vectors can be different depending on what you want to do with your vector. Knowing that will, hopefully, help you with your future ML projects.
### What is a vector?
Previously we defined a vector as a list of numbers or a 1-D data structure. This helped us understand how we could encode information from our datasets as inputs to pass to our ML models. 
Within these models, once that input is received, it’s important to understand that vectors are objects which have the unusual distinction of having both a magnitude and a direction.

### Outputs: Using vector operations 
As we have been noting throughout this post, how you use vectors in ML changes depending on the pipeline stage. In the last section, we saw that vector math, operations, and transformations are key to understanding what’s going on “under the hood” of deep learning neural networks. These computations are all taking place in the “hidden layers” between the input and the output. 

But, for most ML projects, you won’t need this level of detail. Sure, it’s definitely good to understand the maths at the heart of deep learning algorithms, but it’s not critical to getting started with these models. 

For example, as we showed in the input section, you don’t need detailed knowledge of vector maths to be able to encode inputs in a vector format. The magnitude of your vector isn’t important when you’re figuring out how best to encode a sentence so that your model can process it. Instead, it’s important to know the problem you’re solving with encoding, whether you need a sparse or dense vector, or what features you want to capture. 

And now, similarly, in relation to the output of your model, we’ll look at the aspect of vectors which is most impactful for this stage of the ML pipeline. Remember, the output of these models might not even be a vector, it could be an image, or some texts, or a category, or a number like a house price. In these cases you don’t have any vectors to deal with, so you can carry on as before.

But in case you do have a vector output, you will primarily be concerned with two goals:

Input to other models: You may want to use the vector output of one model as the input to another model, like we did with the USE and our Bowie model earlier. In these cases you can refer to the input section, and think of the vector as a list of values and an encoding that represents the information in question. You may also use the output to build a classifier on top of this input, and train it to differentiate between some domain specific data. Either way, you’re using the vector as an input into another model. 
Input to vector functions: If the output is a vector and we’re not using it as an input in another model, then we need to use it in conjunction with a vector function. The USE model outputs an embedding (i.e. a vector), but we can’t interpret this 512 array of numbers in isolation. We need to perform some function on it to generate a result which we can interpret. These functions, as we mentioned in the last section, can do things like identify similarity between two vectors, and reduce the dimensions of a vector so we can visualize them. It can be confusing to know which operation you need for a given purpose. As a result, we will look at some worked examples of these operations in this section.




## Machine Learning with Python + SciKitLearn

Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language.[2] It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.

It is licensed under a permissive simplified BSD license and is distributed under many Linux distributions, encouraging academic and commercial use.

The library is built upon the SciPy (Scientific Python) that must be installed before you can use scikit-learn. This stack that includes:

NumPy: Base n-dimensional array package
SciPy: Fundamental library for scientific computing
Matplotlib: Comprehensive 2D/3D plotting
IPython: Enhanced interactive console
Sympy: Symbolic mathematics
Pandas: Data structures and analysis
Extensions or modules for SciPy care conventionally named SciKits. As such, the module provides learning algorithms and is named scikit-learn.

The vision for the library is a level of robustness and support required for use in production systems. This means a deep focus on concerns such as easy of use, code quality, collaboration, documentation and performance.

Although the interface is Python, c-libraries are leverage for performance such as numpy for arrays and matrix operations, LAPACK, LibSVM and the careful use of cython.

### Who is using Scikitlearn?
The scikit-learn testimonials page lists Inria, Mendeley, wise.io , Evernote, Telecom ParisTech and AWeber as users of the library.

If this is a small indication of companies that have presented on their use, then there are very likely tens to hundreds of larger organizations using the library.

It has good test coverage and managed releases and is suitable for prototype and production projects alike.

### Scikitlearn Resources
If you are interested in learning more, checkout the Scikit-Learn homepage that includes documentation and related resources.

You can get the code from the github repository, and releases are historically available on the Sourceforge project.

### Scikitlearn Documentation
I recommend starting out with the quick-start tutorial and flicking through the user guide and example gallery for algorithms that interest you.

Ultimately, scikit-learn is a library and the API reference will be the best documentation for getting things done. https://scikit-learn.org/stable/#
