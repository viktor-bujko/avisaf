## Avisaf
Avisaf is a tool used for **creating**, **training** and **testing** spaCy-based
Named Entity Recognition models in Aviation Safety Reports.

### Installation 

Avisaf is now available primarily for Unix-like OS:

Please run `$ ./install.sh` script which installs the application and also 
downloads spaCy default English language model.
     
### How to launch the application

   * Application can be started by typing `avisaf` + one of the following 
     commands: `test`, `train`, `autobuild`, `build`.  
     Example: `avisaf test [additional arguments]`
    
   * Feel free to add `-h` parameter anytime if you're in doubt about how
     the program should be used.

### Documentation

Read more details about avisaf in the documentation available [here].

[here]: www.google.com
     
### Program description

The program consists of 3 different stages of NER model creation:

    * avisaf [auto]build
    * avisaf train
    * avisaf test
    
#### avisaf \[auto]build

#### avisaf train

#### avisaf test

### Obtained results

The results can be checked either by checking the recognition accuracy `stats checker`
or by `avisaf test` and the result visualization.

### Examples

![NER example 1](results/example1.png)

![NER example 2](results/example2.png)

&copy; Viktor Bujko 2020

