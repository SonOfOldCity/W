from dynamic_programming import dynamic_programming

class Knap_Prob:
    def __init__(self):
        self.kpcplex = None
        self.kd = None
        self.factor = []
        self.c = []
        self.lhs = []
        self.rhs = []
        self.rows = []
        self.sense = []

    def reinit(self, kd, actions):
        try:
            self.kd = kd
            self.factor = actions
            number = self.kd.items_num
            capacity = self.kd.knap_cap
            weight_cost = [(self.kd.items[i][0],
                            (self.factor[0] * self.kd.items[i][1]
                             + self.factor[1] * self.kd.items[i][2]
                             + self.factor[2] * self.kd.items[i][3])) for i in range(self.kd.items_num)]
            best_cost, best_combination =dynamic_programming(number,capacity,weight_cost)
            best_combination_str = " ".join("%s" % i for i in best_combination)
            print("%s %s  %s\n" % (number, best_cost, best_combination_str))
        except Exception as e:
            print(e)
        # self.kpcplex = cplex.Cplex()
        # self.kd = kd
        # self.factor = actions
        # self.c = [self.factor[0] * self.kd.items[i][0]
        #           + self.factor[1] * self.kd.items[i][1]
        #           + self.factor[2] * self.kd.items[i][2] for i in range(self.kd.items_num)]
        # self.lhs = [0]
        # self.rhs = [self.kd.knap_cap]
        # self.rows = [[self.kd.items[j][0] for j in range(self.kd.items_num)] for i in range(len(self.lhs))]
        # self.sense = ['L' for i in range(len(self.rhs))]
        # self.sense = ''.join(self.sense)
        # self.build_model()
        return best_combination, best_cost

    def build_model(self):
        try:
            self.kpcplex.objective.set_sense(self.kpcplex.objective.sense.maximize)
            self.kpcplex.variables.add(types='B'*self.kd.items_num,obj=self.c)
            self.kpcplex.linear_constraints.add(lin_expr=self.rows,senses=self.sense,rhs=self.rhs)
        except Exception as e:
            print(e)

    def solve_model(self):
        self.kpcplex.solve()
        solutions = self.kpcplex.solution.get_values()
        obj_val = self.kpcplex.solution.get_objective_value()

        return solutions, obj_val