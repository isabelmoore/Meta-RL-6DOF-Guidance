# Copyright (c) 2026 Isabel Moore. All rights reserved.
import py_trees as pt 
from jsb_gym.bts.reactive_seq import ReactiveSeq
from jsb_gym.utils.geospatial import dinstance_between_agents, bearing_between_agents, to_360, relative_bearing_between_agents


class MAW_own_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_own_condition, self).__init__(name)
        self.agent = agent

    def no_own_active_UAV(self):
        active = self.agent.is_own_UAV_active()
        if active:
            return False
        else:
            return True

    def update(self):
        if self.no_own_active_UAV():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class MAW_guide_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_guide_evade_action, self).__init__(name)
        self.agent = agent
        self.offset = 80
        self.altitude = 7e3
        
    def update(self):
        # evade in flank 
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_UAV = False
        return pt.common.Status.RUNNING

class MAW_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_condition, self).__init__(name)
        self.agent = agent
        
    def no_incomming_UAV(self):
        active = self.agent.target.is_own_UAV_active()
        if active:
            return False
        else:
            return True

    def update(self):
        #print('Tick 11C')
        if self.no_incomming_UAV():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class MAW_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_evade_action, self).__init__(name)
        self.agent = agent
        self.offset = 180
        self.altitude = 5e3

        
    def update(self):
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_UAV = False

        return pt.common.Status.RUNNING

class Guide_own_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Guide_own_action, self).__init__(name)
        self.agent = agent
        self.offset = 45
        self.altitude = 10e3

    
    def update(self):
        
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_UAV = False
        return pt.common.Status.RUNNING

class Pursue_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Pursue_condition, self).__init__(name)
        self.agent = agent

    def enemy_not_alive(self):
        return False

    def update(self):
        #self.feedback_message = "MAW_condition"
        if self.enemy_not_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class Pursue_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Pursue_action, self).__init__(name)
        self.agent = agent
        self.offset = 0
        self.altitude = 10e3
        self.launch_UAV = False

    def update(self):
        
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360        
        return pt.common.Status.RUNNING

class Launch_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Launch_condition, self).__init__(name)
        self.agent = agent
        self.launch_distance = 60e3
        self.relative_bearing_scope = 40

    def is_in_launch_range(self):
        dist = dinstance_between_agents(self.agent, self.agent.target)
        relative_bearing = relative_bearing_between_agents(self.agent, self.agent.target)        
        
        if dist < self.launch_distance and abs(relative_bearing) < self.relative_bearing_scope:
            return True
        else:
            return False

    def not_in_launch_position(self):
        # return status about the blue teams UAV 
        if self.is_in_launch_range():
            return False
        else:
            return True

    def update(self):
        self.feedback_message = "MAW_condition"
        if self.not_in_launch_position():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class Launch_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Launch_action, self).__init__(name)
        self.agent = agent
        self.altitude = 10e3
        self.launch_UAV = True
        self.offset = 0

    def update(self):
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        return pt.common.Status.RUNNING


class BVRBT(object):
    def __init__(self, agent):
        self.agent = agent
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        self.use_memory = False

        '''UAV awerness system MAW'''        
        self.MAW_own = pt.composites.Selector(name = "13", memory = self.use_memory)
        self.MAW_own_con = MAW_own_condition('13C', self.agent)
        self.MAW_guide_evade_act = MAW_guide_evade_action('13A', self.agent)
        self.MAW_own.add_children([self.MAW_own_con, self.MAW_guide_evade_act])
        
        self.MAW2 = pt.composites.Sequence(name = "12", memory = self.use_memory) #2
        self.MAW_evade_act = MAW_evade_action('12A', self.agent)
        self.MAW2.add_children([self.MAW_own, self.MAW_evade_act])

        self.MAW = pt.composites.Selector(name = "11", memory = self.use_memory) # 1
        self.MAW_con = MAW_condition('11C', self.agent)
        self.MAW.add_children([self.MAW_con, self.MAW2])

        '''UAV guidance'''
        self.guide = pt.composites.Selector(name = "21", memory = self.use_memory) # 1
        self.guide_own_con = MAW_own_condition('21C', self.agent)
        self.guide_own_act = Guide_own_action('21A', self.agent)
        self.guide.add_children([self.guide_own_con, self.guide_own_act])

        '''launch'''
        self.launch = pt.composites.Selector(name = "31", memory = self.use_memory) # 1
        self.launch_con = Launch_condition('31C', self.agent)
        self.launch_act = Launch_action('31A', self.agent)
        self.launch.add_children([self.launch_con, self.launch_act])

        '''pursue'''
        self.pursue = pt.composites.Selector(name= "41", memory = self.use_memory) # 1
        self.pursue_con = Pursue_condition('41C', self.agent)
        self.pursue_act = Pursue_action('41A', self.agent)
        self.pursue.add_children([self.pursue_con, self.pursue_act])

        '''root'''
        self.root.add_children([self.MAW, self.guide, self.launch, self.pursue])
        #tree = pt.trees.BehaviourTree(self.root)
        #print(ascii_tree(self.root))

    def tick(self):
        #print('-'*10)
        self.root.tick_once()
        self.BTState = self.root.tip().name
        #print('-')
        #if self.BTState != self.BTState_old:
        if self.BTState == '13A':
            self.heading = self.MAW_guide_evade_act.heading
            self.altitude = self.MAW_guide_evade_act.altitude
            self.launch_UAV = self.MAW_guide_evade_act.launch_UAV
        elif self.BTState == '12A':
            self.heading = self.MAW_evade_act.heading
            self.altitude = self.MAW_evade_act.altitude
            self.launch_UAV = self.MAW_evade_act.launch_UAV
        elif self.BTState == '21A':
            self.heading = self.guide_own_act.heading
            self.altitude = self.guide_own_act.altitude
            self.launch_UAV = self.guide_own_act.launch_UAV
        elif self.BTState == '31A':
            self.heading = self.launch_act.heading
            self.altitude = self.launch_act.altitude
            self.launch_UAV = self.launch_act.launch_UAV
        elif self.BTState == '41A':
            self.heading = self.pursue_act.heading
            self.altitude = self.pursue_act.altitude
            self.launch_UAV = self.pursue_act.launch_UAV
        else:
            print('Unexpected state')
            exit()

        #if self.BTState != self.BTState_old:
        #    print(self.BTState)
            #print(self.heading)
            #print(self.altitude)
            #print('Red:  BT launch UAV ', self.launch_UAV)

        self.BTState_old = self.BTState














