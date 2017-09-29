# Tony

My first personal intelligent assistant - An intelligent resume.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to install packages mentioned in the requirements.txt to run the program. Run the following in a bash terminal to do so.

```
$virtualenv venv
$source venv/bin/activate
$pip install -r requirements.txt
```

### Installation

First, you need to have git, python2.7 and python-pip installed. For debian-based distributions use the command below in a bash terminal.

```
$sudo apt install git python2.7 python-pip
```

Now, you should clone the repository.

```
$git clone https://github.com/Razzaghnoori/Tony 
```

Next, entre the folder created, create virtual environment, install prerequisits and initialize the program using the following commands (note that the last line may take a while to complete):

```
$cd Tony

$virtualenv venv
$source venv/bin/activate
$pip install -r requirements.txt

$python initialize.py
```

Finally, run the following command to start the program.

```
$python main.py
```

## Authors

* **Mohammad Razzaghnoori** - *Initial work* - [Mohammad ArEn](https://github.com/razzaghnoori)

