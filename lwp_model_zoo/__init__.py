from torch import hub
hub._validate_not_a_forked_repo = lambda a, b, c: True
