from random import randint, uniform


def generate_valid_data():
    data = dict(
        age=randint(0, 125),
        sex=randint(0, 1),
        cp=randint(0, 3),
        trestbps=randint(0, 200),
        chol=randint(100, 600),
        fbs=randint(0, 1),
        restecg=randint(0, 2),
        thalach=randint(0, 250),
        exang=randint(0, 1),
        oldpeak=uniform(0, 10),
        slope=randint(0, 2),
        ca=randint(0, 3),
        thal=randint(0, 2),
    )
    return data


def generate_invalid_data():
    data = dict(
        age=randint(-100, -1),
        sex=randint(3, 4),
        cp="string here",
        trestbps=uniform(0, 200),
        chol=uniform(100, 600),
        fbs=uniform(0, 1),
        restecg=uniform(0, 2),
        thalach=uniform(0, 250),
        exang=randint(0, 1),
        oldpeak=uniform(0, 10),
        slope=randint(0, 2),
        ca=[1, 2, 3],
        thal=randint(20, 28),
    )
    return data