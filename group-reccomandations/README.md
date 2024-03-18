# Group Recommendations System

This project implements a group recommendation system using user-based collaborative filtering, implemented in Python.
(Uses the [MovieLens Small dataset](https://grouplens.org/datasets/movielens/latest/))

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Demo via Command Line](#demo-via-command-line)
  - [Web Application](#web-application)
- [File Description](#file-description)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the project dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

### Demo via Command Line

To run the program via command line and get group recommendations:

```
python demo.py
```

### Web Application

To start the web application:

```
python webapp.py
```

This will start a web application on localhost port 5000. Open a web browser and visit [http://localhost:5000](http://localhost:5000) to use the application.

## File Description

- `demo.py`: Script to run the group recommendation system via command line.
- `webapp.py`: Script to start a web application for the recommendation system.

## Dependencies

The project dependencies are listed in the `requirements.txt` file.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
