class Clock:
    """
    The Clock class is responsible for keeping track of simulation time and regulates updates to the control policy
    and to the rendered scene
    """

    def __init__(self, sim_rate, controller_rate=100, render_rate=24):
        """
        :param sim_rate: simulation rate (Hz)
        :param controller_rate: controller rate (Hz)
        :param render_rate: render rate (Hz)
        """
        self.reset_rates(sim_rate, controller_rate, render_rate)
        self.current_step = 0
        self.current_time = 0.0
        self.last_tick = 0.0
        self.timer_start = [0., 0., 0., 0., 0.] # space for 5 timers

    def reset_rates(self, sim_rate=None, controller_rate=None, render_rate=None):
        if sim_rate is not None: self.sim_rate = sim_rate
        if controller_rate is not None: self.controller_rate = controller_rate
        if render_rate is not None: self.render_rate = render_rate

        self.sim_timestep = 1 / self.sim_rate
        self.relative_controller_rate = int(self.sim_rate / self.controller_rate)
        self.relative_render_rate = int(self.sim_rate / self.render_rate)

    def step(self):
        """
        Advance the simulation time by one simulation timestep
        """
        self.current_step += 1
        self.current_time += self.sim_timestep

    def time(self):
        """
        Get current simulation time
        """
        return self.current_time

    def tick(self, interval=1.0):
        """
        Called in a loop, returns True every <interval> seconds
        Useful to schedule operations (e.g. debug prints) every N seconds
        :rtype: bool
        """
        if self.current_time - self.last_tick > interval:
            self.last_tick = self.current_time
            return True
        else:
            return False

    def set_timer(self, index=0):
        """
        Start a timer at this moment
        """
        self.timer_start[index] = self.current_time

    def get_timer(self, interval=1.0, index=0):
        """
        check if <interval> seconds have passed since set_timer was called
        :rtype: bool
        """
        return self.current_time - self.timer_start[index] > interval

    def is_controller_update(self):
        return self.current_step % self.relative_controller_rate == 0

    def is_render_update(self):
        return self.current_step % self.relative_render_rate == 0