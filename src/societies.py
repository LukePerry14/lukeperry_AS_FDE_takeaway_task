from __future__ import annotations

from dataclasses import Field, dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

INCOME = ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000", "$75,000-$100,000", "$100,000 or more"]
AGE = ["18-29", "30-49", "50-64", "65+"]
RACE = ["White", "Black", "Hispanic", "Asian", "Other"]

@dataclass
class DemoAttr:
    """
    Dataclass for storage of demographic attributes
    """
    id: Optional[int] = None
    income: Optional[str] = None
    age: Optional[str] = None
    race: Optional[str] = None


class Society:
    """
    Acts as a wrapper on the 100 simulated People, storing their information and orchestrating society level operations
    """
    def __init__(self, attrbs: List[DemoAttr], questions: Dict[str, Dict]):
        self.people = [Person(attrb) for attrb in attrbs]
        self.questions = questions
        self.QAs: Dict[str, List[Dict[str, float]]] = {}

        self.adj = [np.array([p1 * p2 for p2 in self.people], dtype=float) for p1 in self.people]

        for i in range(len(self.people)):
            self.adj[i][i] = 0
            self.adj[i] = self.adj[i] / self.adj[i].sum()

    def record(self, qkey: str, responses: List[Dict[str, float]]):
        self.QAs[qkey] = responses

    def build_backstories(self, social: bool):
        from .model_handling import BackstoryGenerator
        from tqdm import tqdm

        gen = BackstoryGenerator()

        for person in tqdm(self.people, desc="Persona backstories generated"):
            person.backstory = gen.generate(person, None)
            if social and self.adj:

                neighbours = np.random.choice(self.people, 2, p=self.adj[person.person_id]).tolist()
                person.social_backstory = gen.generate(person, neighbours)


    def evaluate(self, qkey: str, tranches: bool):
        """
        Aggregate persona responses and return normalised Wasserstein distance vs. ground truth.
        Personas can be treated as population tranches, with all response distributions averaged,
        or they can be treated as individuals from that tranch, sampling from the produced distribution.
        """
        
        # Persona Responses
        person_responses = self.QAs[qkey]
        q = self.questions[qkey]


        options = q["options"][:-1]
        ground_truth = q["responses"]

        ordinal = np.array(q["ordinal"], dtype=float)

        # Mean probability per option across all 100 personas
        if tranches:
            agg = [
                sum(r[opt] for r in person_responses) / len(person_responses)
                for opt in options
            ]
        else:  # sample a discrete answer per person, generate distribution from these answers
            answers = [np.random.choice(options, p=np.array([r[opt] for opt in options])) for r in person_responses]
            agg = [answers.count(opt) / len(answers) for opt in options]

        wd = wasserstein_distance(ordinal, ordinal, ground_truth, agg) / (ordinal.max() - ordinal.min())
        return wd, agg


class Person:
    """
    Acts as a wrapper on a set of attributes, 
    """

    def __init__(self, attrb: DemoAttr):
        self.attrb = attrb

        self.person_id = attrb.id

        self.latent = (INCOME.index(attrb.income) / (len(INCOME) - 1), AGE.index(attrb.age) / (len(AGE) - 1), RACE.index(attrb.race))


    def __mul__(self, other):
        """
        replace wth latent space distance passed through sigmoid function to produce link probability.
        Scale each distance by value so that average distance in each dimension is 1.
        """
        d_inc = abs(self.latent[0] - other.latent[0]) * 2
        d_age = abs(self.latent[1] - other.latent[1]) * 1.8
        d_race = 0 if (self.latent[2] == other.latent[2]) else 1.25

        d_tot = (d_inc + d_age + d_race) / 3

        return 1 / (1 + np.exp(3 * (d_tot - 0.84)))  

    def attrb_string(self):
        """
        Generates System persona creation string based on demographic information and optionally demographic information of social network peers.
        """
        if not self.attrb:
            raise ValueError("Not yet instantiated attrb for this Person")

        characteristics = "You are a member of the American public with the following demographic characteristics:\n"
        
        attrb = self.attrb

        demographic_fields = [
            ("Age", attrb.age),
            ("Income Level", attrb.income),
            ("Race", attrb.race),
        ]
        
        for label, value in demographic_fields:
            if value is not None:
                characteristics += "- {}: {}\n".format(label, value)
        
        
        return characteristics
    

    def attribute_list(self):
        
        if not self.attrb:
            raise ValueError("Not yet instantiated attrb for this Person")

        cls_fields: Tuple[Field, ...] = fields(DemoAttr)

        attributes = []

        for field in cls_fields:
            attributes.append(getattr(self.attrb, field.name))

        return tuple(attributes)