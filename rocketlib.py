import numpy as np
import pylab as pl

g0 = 9.807 # m/s²

def to_polar(x, y):
    return np.array((np.sqrt(x**2 + y**2), np.angle(x + 1j * y)))

def to_cart(norm, angle):
    return np.array((norm * np.cos(angle), norm*np.sin(angle)))

def compute_mdot(thrust_vac, Ispvac):
    # thrust_vac in N
    # Ispvac in s
    return thrust_vac / (Ispvac * g0) 

# https://en.wikipedia.org/wiki/Ariane_5
# https://core.ac.uk/download/pdf/77231853.pdf
# https://www.ariane.group/wp-content/uploads/2020/06/VULCAIN2.1_2020_04_PS_EN_Web.pdf
# http://www.astronautix.com/
ariane_stage1_EAP_P241 = { # equipped with vulcain 2 engine
    'diameter':5.4, # used to compute drag
    'nozzle_exit_diameter':2.1,
    'Ispvac':432, 
    'dry_mass':14.7e3, 
    'thrust_vac':1390e3
    #'mdot'=323
}

ariane_stage2_ESC_A = { # 
    'diameter':0, # used to compute drag, but already considered in stage 1
    'nozzle_exit_diameter':0.99,
    'Ispvac':446, 
    'dry_mass':4.54e3, 
    'thrust_vac':67e3
    #'mdot'=323
}

ariane_booster_EPC_H173 = {
    'diameter':3.06, # used to compute drag
    'nozzle_exit_diameter':3.06,
    'Ispvac':275, 
    'dry_mass':33e3,
    'thrust_vac':7080e3
}

ariane_stage1 = ariane_stage1_EAP_P241
ariane_stage2 = ariane_stage2_ESC_A
ariane_booster = ariane_booster_EPC_H173

class Engine():
    
    def __init__(self, propelant_mass, model=ariane_stage1):
        
        # default from Vulcain 2: https://en.wikipedia.org/wiki/Vulcain_(rocket_engine) 
        # and https://arc.aiaa.org/doi/10.2514/1.A33363
        
        self.started = False
        self.Adrag = model['diameter']**2 * np.pi / 4
        self.Ae = model['nozzle_exit_diameter']**2 * np.pi / 4
        self.Ispvac = model['Ispvac'] # s
        self.mdot = compute_mdot(model['thrust_vac'], model['Ispvac']) # kg/s
        self.propelant_mass = propelant_mass
        self.dry_mass = model['dry_mass']
    
    def is_started(self): 
        return bool(self.started)
    
    def start(self):
        if self.is_started(): print('warning: engine already started')
        else: self.started = True

    def thrust(self, dt, p0=101325):
        if not self.is_started(): return 0
        if self.is_empty(): return 0
        # dt = thrusting time in s
        # p0 = external ambiant pressure in Pa
        
        # get only a fraction of thrust from what's left of propelant
        needed_mass = self.mdot * dt
        ratio = min(1, self.propelant_mass / needed_mass)
        self.propelant_mass -= needed_mass * ratio
        
        # https://en.wikipedia.org/wiki/Rocket_engine_nozzle
        F = self.Ispvac * g0 * self.mdot - self.Ae * p0 # F in N
        
        return F * ratio
    
    def drag(self, dt, density, velocity, coefficient=0.7):
        # https://www.grc.nasa.gov/WWW/k-12/rocket/drageq.html
        # https://www.grc.nasa.gov/www/k-12/rocket/shaped.html
        drag = coefficient * density * velocity**2 * self.Adrag / 2
        return drag
    
    def get_mass(self):
        return self.dry_mass + self.propelant_mass
    
    def is_empty(self):
        if self.propelant_mass > 0: return False
        return True
    
    
class Planet():
    
    def __init__(self, mass=5.972168e24, radius=6371e3):
        # default Earth
        self.mass = mass # kg
        self.radius = radius # m
        
    def get_g(self, p):
        # p = (x,y) : coordinates wrt to launch position on the surface at pole (altitude=0, angle=90)
        G = 6.67430e-11
        r, angle = self.get_polar(p)
        #angle -= np.pi/2
        altitude = r - self.radius
        
        g = - G * self.mass / ((altitude + self.radius)**2)
        #print(g, altitude, altitude + self.radius, angle)
        return to_cart(g, angle)
    
    def get_polar(self, p):
        # x,y : coordinates wrt to launch position on the surface at pole (altitude=0, angle=pi/2)
        return to_polar(p[0], p[1] + self.radius)
    
    def get_altitude(self, p):
        return self.get_polar(p)[0] - self.radius
    
    
    def get_parameters(self, altitude):
        # https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
        if altitude < 11000:
            T = 15.05 - 0.00649 * max(altitude, 0)
            p = 101.29 * ((T + 273.15) / 288.08)**(5.256)
        elif altitude >= 11000 and altitude < 25000:
            T = -56.46
            p = 22.65 * np.exp(1.73 - 0.000157 * altitude)
        else:
            T = -131.21 + 0.00299 * altitude
            p = 2.488 * ((T + 273.15) / 216.6)**(-11.388)
        return T, p
    
    def get_pressure(self, altitude):
        return self.get_parameters(altitude)[1]
    
    def get_density(self, altitude):
        # https://en.wikipedia.org/wiki/Density_of_air
        Rspec = 287.0500676 # J/kg/K
        T, p = self.get_parameters(altitude)
        return p / (Rspec * (T + 273.15))


class Rocket():
    
    def __init__(self, altitude=1, angle=90, dry_mass=1, dt_min=1, engines=None, planet=None, dt_fixed=False):
        self.engines = dict() # dictionnary of attached engines
        self.all_engines = list() # list of all engines (including detached ones)
        self.events = list()
        
        if planet is None:
            self.planet = Planet() # default Earth
        
        self.t = 0
        self.time_index = 0
        self.dt_min = dt_min # s
        self.dt = self.dt_min
        self.dt_fixed = bool(dt_fixed)
        self.p = np.array((0, altitude), dtype=float) # m
        self.angle = np.deg2rad(angle)
        self.thruster_angle = 0
    
        assert dry_mass > 0, 'dry_mass cannot be null'
        self.dry_mass = dry_mass
        
        
        self.v = np.array((0, 0), dtype=float)
        self.a = np.array((0, 0), dtype=float)
        self.f = np.array((0, 0), dtype=float)
        
        self.log = {
            't': list(),
            'p': list(),
            'v': list(),
            'a': list(),
            'mass': list(),
            'altitude': list(),
            'f_thrust':list(),
            'f_grav':list(),
            'f_drag':list(),
            'angle':list(),
            'pressure':list(),
            'density':list()
                   }
        
        
    def attach_engine(self, name, propelant_mass, model):
        assert name not in self.engines, 'engine already exists'
        self.engines[name] = Engine(propelant_mass, model=model)
        self.all_engines.append(name)
        
    def detach_engine(self, name):
        del self.engines[name]
        
    def is_detached(self, name):
        if name not in self.all_engines:
            raise Exception('error: {} never attached')
        if name in self.engines:
            return False
        return True

    def is_started(self, name):
        if self.is_detached(name):
            return True
        return self.engines[name].is_started()
        
    def start_engine(self, name):
        if self.engines[name].is_started(): return
        self.engines[name].start()
        self.append_event('{} started'.format(name))
        
    def set_thruster_angle_wrt_axis(self, angle):
        # thruster angle wrt rocket vertical axis
        self.thruster_angle = np.deg2rad(angle)
        
    def set_thruster_angle_wrt_gravity(self, angle):
        # thruster angle wrt gravity angle
        angle = np.deg2rad(angle)
        _, grav_angle = to_polar(*self.planet.get_g(self.p))
        
        self.thruster_angle = (grav_angle + np.pi + angle) - self.get_angle()
        
    def set_velocity(self, vel, angle):
        self.v = to_cart(vel, np.deg2rad(angle))
        
    def set_acceleration(self, acc, angle):
        self.f = to_cart(acc * self.mass, np.deg2rad(angle))        
        
    def get_angle(self):
        pol = to_polar(*self.v)
        if pol[0] == 0: 
            return self.angle
        self.angle = pol[1]
        
        return self.angle
    
    def get_angle_wrt_gravity(self):
        # thruster angle wrt gravity angle
        return to_polar(*self.planet.get_g(self.p))[1] + np.pi - self.angle
    
    def get_altitude(self):
        return self.planet.get_altitude(self.p)
    
    def append_event(self, name):
        self.events.append((name, self.time_index, self.t))
        print('({:.1e}s): {}'.format(self.t, name))
        
    def check_engines(self):
        for ieng in list(self.engines.keys()):
            if self.engines[ieng].is_empty():
                self.detach_engine(ieng)
                self.append_event('{} detached'.format(ieng))
        
    def get_mass(self):
        self.check_engines()
        mass = self.dry_mass
        for ieng in list(self.engines.keys()):
            mass += self.engines[ieng].get_mass()
        return mass
    
    def update(self):
        self.mass = self.get_mass() # check engines and update mass
        
        
        if self.dt_fixed:
            self.dt = self.dt_min
        else:
            self.dt = max(self.dt_min, np.max(np.abs(self.v)) / 5000.)
        
        p0 = self.planet.get_pressure(self.get_altitude())
        rho0 = self.planet.get_density(self.get_altitude())
        
        f_thrust = 0
        f_drag = np.array((0,0), dtype=float)
        
        for ieng in list(self.engines.keys()):
            f_thrust += self.engines[ieng].thrust(self.dt, p0=p0)
            v_norm = np.sqrt(np.sum(self.v**2))
            if v_norm != 0:
                f_drag -= self.engines[ieng].drag(self.dt, rho0, v_norm) * self.v / v_norm
        
        f_thrust = to_cart(f_thrust, self.get_angle() + self.thruster_angle)
        #self.mass = self.get_mass()
        f_grav = self.planet.get_g(self.p) * self.mass
        self.f = f_grav + f_thrust + f_drag
        
        self.a = self.f / self.mass
        self.v += self.a * self.dt
        self.p += self.v * self.dt
        
        # log data
        self.t += self.dt
        self.time_index += 1
        self.log['t'].append(self.t)
        self.log['p'].append(np.copy(self.p))
        self.log['v'].append(np.copy(self.v))
        self.log['a'].append(np.copy(self.a))
        self.log['mass'].append(np.copy(self.mass))
        self.log['altitude'].append(self.get_altitude())
        self.log['f_thrust'].append(f_thrust)
        self.log['f_drag'].append(f_drag)
        self.log['f_grav'].append(f_grav)
        self.log['angle'].append(np.rad2deg(self.get_angle_wrt_gravity()))
        self.log['pressure'].append(p0)
        self.log['density'].append(rho0)
        
        
    def get_log(self, key):
        return np.array(self.log[key])
    
    def plot_vs_time(self, key):
        pl.figure()
        p = self.get_log(key)
        time = self.get_log('t')
        
        if len(p.shape) > 1:
            pl.plot(time, p[:,0], ls='-', lw=1, c='blue')
            #pl.scatter(time, p[:,0], marker='+', c='blue')
            pl.plot(time, p[:,1], ls='-', lw=1, c='orange')
            #pl.scatter(time, p[:,1], marker='+', c='orange')
            pl.plot(time, np.sqrt(p[:,0]**2 + p[:,1]**2), ls='-', lw=1, c='0.')
        else:
            pl.plot(time, p, ls='-', lw=1, c='blue')
            #pl.scatter(time, p, marker='+', c='blue')
        #pl.axis('equal')
        #pl.yscale('log')
        cmap = pl.get_cmap("tab10")
        for ievent, i in zip(self.events, range(len(self.events))):
            pl.axvline(time[ievent[1]], label='({:.1e}s):{}'.format(ievent[2], ievent[0]), c=cmap(i))
        pl.grid()
        pl.title(key)
        if len(self.events) > 0: pl.legend()
        
    def plot_trajectory(self):
        from matplotlib.patches import Circle
        
        pl.figure()
        ratio = 1e3
        
        
        circle = Circle((0, -self.planet.radius / ratio), self.planet.radius / ratio, color='gray', fill=False)
        pl.gca().add_patch(circle)
        
        # geostationary orbit
        circle = Circle((0, -self.planet.radius / ratio), 42164e3 / ratio, color='pink', fill=False)
        pl.gca().add_patch(circle)
        
        p = self.get_log('p')
        x, y = p[:,0] / ratio, p[:,1] / ratio
        pl.plot(x, y, ls='-', lw=1)
        #pl.scatter(x, y, marker='.')
        for ievent in self.events:
            pl.scatter(x[ievent[1]], y[ievent[1]], label='({:.1e}s):{}'.format(ievent[2], ievent[0]))
        thrust = self.get_log('f_thrust')
        thrust /= self.mass * 10 #
        thrust[np.isnan(thrust)] = 0
        pl.quiver(x, y, thrust[:,0], thrust[:,1], angles='xy')
        
        pl.axis('equal')
        
        border = max((np.max(x) - np.min(x), np.max(y) - np.min(y))) * 0.1
        pl.xlim(np.min(x) - border, np.max(x) + border)
        pl.ylim(np.min(y) - border, np.max(y) + border)
        pl.grid()
        pl.title('Trajectory')
        if len(self.events) > 0: pl.legend()
        
        
        