import copy

class State:
    def __init__(self, sleft, wleft, sright = 0, wright = 0, boat = False):
        self.sleft = sleft
        self.sright = sright
        self.wleft = wleft
        self.wright = wright
        self.boat = boat
        self.key = '%s_%s_%s_%s_%s'%(sleft,wleft,sright,wright,boat)
    
    def isMatching(self, another_state):
        return  (self.sleft == another_state.sleft and
        self.sright == another_state.sright and
        self.wleft == another_state.wleft and
        self.wright == another_state.wright and
        self.boat == another_state.boat)

    def isValid(self):
        if (self.sleft != 0 and self.sleft < self.wleft):
            return False
        if (self.sright != 0 and self.sright < self.wright):
            return False
        return True

    def l2w(self):
        return State(self.sleft, self.wleft - 2, self.sright, self.wright + 2, True)

    def l2s(self):
        return State(self.sleft - 2, self.wleft, self.sright + 2, self.wright, True)

    def l1s(self):
        return State(self.sleft - 1, self.wleft, self.sright + 1, self.wright, True)

    def l1w(self):
        return State(self.sleft, self.wleft - 1,  self.sright, self.wright + 1, True)

    def l1w1s(self):
        return State(self.sleft - 1, self.wleft - 1, self.sright + 1, self.wright + 1, True)

    def r2w(self):
        return State(self.sleft, self.wleft + 2, self.sright, self.wright - 2, False)

    def r2s(self):
        return State(self.sleft + 2, self.wleft, self.sright - 2, self.wright, False)

    def r1s(self):
        return State(self.sleft + 1, self.wleft,  self.sright - 1, self.wright, False)

    def r1w(self):
        return State(self.sleft, self.wleft + 1, self.sright, self.wright - 1, False)

    def r1w1s(self):
        return State(self.sleft + 1, self.wleft + 1, self.sright - 1, self.wright - 1, False)


class SemanticNetsAgent:
    def __init__(self):
        # self.state_dict = {}
        self.goal_state = None
        self.answers = []
        # self.count = 0

    def recurisve_dfs(self, state, state_dictionary, moves, boat, count):
        """ Return moves
        state: current state
        state_dictionary: current state dictionary
        moves: move array made so far
        boat: the direction of the boat
        count: for debugging
        """

        # print(count,state.key,moves,state_dictionary.keys(),self.goal_state.key == state.key)
        # if state match goal state: return moves
        if state.key == self.goal_state.key:
            self.answers.append(moves)
            return True

        # if the state is not valid or we have already been down this path: return
        if(not state.isValid() or state.key in state_dictionary):
            return None
        # else add to dict
        state_dictionary[state.key] = (state, moves)

        copied_dict = copy.deepcopy(state_dictionary)
        # get next states and go through them
        if(boat):
            if(state.wright >= 2):
                r2w = state.r2w()
                r2w_moves = [x for x in moves]
                r2w_moves.append((0,2))
                next = self.recurisve_dfs(r2w, copied_dict, r2w_moves, not boat, count + 1)


            if(state.sright >= 2):
                r2s = state.r2s()
                r2s_mv = [x for x in moves]
                r2s_mv.append((2, 0))
                next = self.recurisve_dfs(r2s, copied_dict, r2s_mv, not boat, count + 1)


            if(state.wright >= 1):
                next_s = state.r1w()
                next_s_moves = [x for x in moves]
                next_s_moves.append((0,1))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)


            
            if(state.sright >= 1):
                next_s = state.r1s()
                next_s_moves = [x for x in moves]
                next_s_moves.append((1,0))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)

            
            if(state.wright >= 1 and state.sright >= 1):
                next_s = state.r1w1s()
                next_s_moves = [x for x in moves]
                next_s_moves.append((1,1))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)


        else:
            if(state.wleft >= 2):
                next_s = state.l2w()
                next_s_moves = [x for x in moves]
                next_s_moves.append((0,2))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)

            
            if(state.sleft >= 2):
                next_s = state.l2s()
                next_s_moves = [x for x in moves]
                next_s_moves.append((2, 0))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)


            if(state.wleft >= 1):
                next_s = state.l1w()
                next_s_moves = [x for x in moves]
                next_s_moves.append((0,1))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)
            
            if(state.sleft >= 1):
                next_s = state.l1s()
                next_s_moves = [x for x in moves]
                next_s_moves.append((1,0))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)

            if(state.sleft >= 1 and state.wleft >= 1):
                next_s = state.l1w1s()
                next_s_moves = [x for x in moves]
                next_s_moves.append((1,1))
                next = self.recurisve_dfs(next_s, copied_dict, next_s_moves, not boat, count + 1)

        return None

    # Recursive dfs even with the changes to get all answers is not good. its not good because we are going all the way down, and then all the way up. too much memory, too much time
    def bfs(self):
        return

    def solve(self, initial_sheep, initial_wolves):
        """Solve Sheep and Wolves if possible"""
        #Add your code here! Your solve method should receive
        #the initial number of sheep and wolves as integers,
        #and return a list of 2-tuples that represent the moves
        #required to get all sheep and wolves from the left
        #side of the river to the right.
        #
        #If it is impossible to move the animals over according
        #to the rules of the problem, return an empty list of
        #moves.

        # if problem is not valid.
        if initial_sheep < initial_wolves:
            return []

        # this is to easy to wait on recurision
        if(initial_sheep == 1 and initial_wolves == 1):
            return [(1,1)]
        
        # reset dict
        intial_state_dict = {}

        # Initialize
        boat = False # boat will be False for left and True for right
        initial_state = State(initial_sheep, initial_wolves)
        goal_state = State(0, 0, initial_sheep, initial_wolves, True)

        self.goal_state = goal_state
        intial_state_dict[goal_state.key] = goal_state
        
        # ----------------------------- bfs ----------------------------
        
        
        
        
        
        # ----------------------------- dfs recursion ----------------------------

        # # Start recursion
        # self.answers = []
        # self.recurisve_dfs(initial_state, intial_state_dict, [], boat, 0)

        # # if none: return empty move array
        # if len(self.answers) < 1:
        #     return []

        # # print('answer_list', self.answers)
        # min_answer = (10000000000000, None)
        # for answer in self.answers:
        #     length = len(answer)
        #     if length < min_answer[0]:
        #         min_answer = (length, answer)

        # # print(min_answer)
        # return min_answer[1]
