#!/usr/bin/python3
# ------------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
#
# Copyright (C) 2018-2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------
""" Swarming parameter search """

# TODO: Deal with global constants: particle_strength, global_strength, velocity_strength
#       Maybe make them into CLI Arguments?

particle_strength   =  .25
global_strength     =  .50
velocity_strength   =  .95
assert(velocity_strength + particle_strength / 2 + global_strength / 2 >= 1)

import argparse
import sys
import os
import random
import pprint

from .nupic.optimization.parameter_set import ParameterSet

class ParticleSwarmOptimizations:
    def __init__(self, lab, args):
        # Setup the particle swarm.
        self.particles     = args.particles
        self.next_particle = random.randrange(args.particles)
        self.swarm_path    = os.path.join(lab.ae_directory, 'swarm')
        try:
            with open(self.swarm_path, 'r') as swarm_file:
                swarm_raw = swarm_file.read()
        except FileNotFoundError:
            # Initialize a new particle swarm.
            self.swarm_data = {}
            for particle in range(self.particles):
                if particle in [0, 1, 2]:
                    # Evaluate the default parameters a few times, before branching out
                    # to the more experimential stuff.  Several evals are needed since
                    # these defaults may have their random velocity applied.
                    value = lab.default_parameters
                else:
                    value = ParameterSet( initial_parameters(lab.default_parameters))
                self.swarm_data[particle] = {
                    'value':      value,
                    'velocity':   initial_velocity(lab.default_parameters),
                    'best':       value,
                    'best_score': None,
                    'hash':       hash(value),
                }
            self.swarm_data['best']       = random.choice(list(self.swarm_data.values()))['best']
            self.swarm_data['best_score'] = None
            self.swarm_data['evals']      = 0
        else:
            # Load an existing particle swarm.
            try:
                self.swarm_data = eval(swarm_raw)
            except SyntaxError:
                while True:
                    print("Corrupted particle swarm data file.  [B]ackup, [O]verwrite, or [EXIT]?")
                    choice = input().upper()
                    if choice == 'B':
                        backup_path = self.swarm_path + ".backup"
                        os.rename(self.swarm_path, backup_path)
                        print("BACKUP PATH: %s"%backup_path)
                        self.swarm_data = initialize_particle_swarm(lab.default_parameters, self.particles)
                        break
                    elif choice == 'O':
                        self.swarm_data = initialize_particle_swarm(lab.default_parameters, self.particles)
                        break
                    elif choice in 'EXITQ':
                        print("EXIT")
                        sys.exit()
                    else:
                        print('Invalid input "%s".'%choice)

            if self.particles != sum(isinstance(key, int) for key in self.swarm_data):
                print("Warning: argument 'particles' does not match number of particles stored on file.")

    def __call__(self, lab):
        # Run the particle swarm optimization.
        particle_data = self.swarm_data[self.next_particle]
        self.next_particle = (self.next_particle + 1) % self.particles

        # Update the particles velocity.
        particle_data['velocity'] = update_particle_velocity(
            particle_data['value'],
            particle_data['velocity'],
            particle_data['best'],
            self.swarm_data['best'],)

        # Update the particles postition.
        particle_data['value'] = update_particle_position(
            particle_data['value'],
            particle_data['velocity'])

        # Evaluate the particle.
        promise = pool.apply_async(evaluate_particle, (particle_data,))

        return parameters

    def collect(self, results):
        particle_data = self.swarm_data[particle_number]
        try:
            score = promise.get()
        except (ValueError, MemoryError, ZeroDivisionError, AssertionError) as err:
            print("")
            print("Particle Number %d"%particle_number)
            pprint.pprint(particle_data['value'])
            print("%s:"%(type(err).__name__), err)
            print("")
            # Replace this particle.
            particle_data['velocity'] = initial_velocity(default_parameters)
            if particle_data['best_score'] is not None:
                particle_data['value'] = particle_data['best']
            elif self.swarm_data['best_score'] is not None:
                particle_data['value'] = self.swarm_data['best']
            else:
                particle_data['value'] = initial_parameters(default_parameters)
            continue
        except Exception:
            print("")
            pprint.pprint(particle_data['value'])
            raise

        # Update best scoring particles.
        if particle_data['best_score'] is None or score > particle_data['best_score']:
            particle_data['best']       = particle_data['value']
            particle_data['best_score'] = score
            print("New particle (%d) best score %g"%(particle_number, particle_data['best_score']))
        if self.swarm_data['best_score'] is None or score > self.swarm_data['best_score']:
            self.swarm_data['best']       = typecast_parameters(particle_data['best'], parameter_structure)
            self.swarm_data['best_score'] = particle_data['best_score']
            self.swarm_data['best_particle'] = particle_number
            print("New global best score %g"%self.swarm_data['best_score'])

        # Save the swarm to file.
        self.swarm_data['evals'] += 1
        with open(swarm_path, 'w') as swarm_file:
            print('# ' + ' '.join(sys.argv), file=swarm_file) # TODO: Get this from lab-report object.
            pprint.pprint(self.swarm_data, stream = swarm_file)


def initial_parameters(default_parameters):
    # Recurse through the parameter data structure.
    if isinstance(default_parameters, dict):
        return {key: initial_parameters(value)
            for key, value in default_parameters.items()}
    elif isinstance(default_parameters, tuple):
        return tuple(initial_parameters(value) for value in default_parameters)
    # Calculate good initial values.
    elif isinstance(default_parameters, float):
        return default_parameters * 1.25 ** (random.random()*2-1)
    elif isinstance(default_parameters, int):
        if abs(default_parameters) < 10:
            return default_parameters + random.choice([-1, 0, +1])
        else:
            initial_value_float = initial_parameters(float(default_parameters))
            return int(round(initial_value_float))

def initial_velocity(default_parameters):
    # Recurse through the parameter data structure.
    if isinstance(default_parameters, dict):
        return {key: initial_velocity(value)
            for key, value in default_parameters.items()}
    elif isinstance(default_parameters, tuple):
        return tuple(initial_velocity(value) for value in default_parameters)
    # Calculate good initial velocities.
    elif isinstance(default_parameters, float):
        max_percent_change = 10
        uniform = 2 * random.random() - 1
        return default_parameters * uniform * (max_percent_change / 100.)
    elif isinstance(default_parameters, int):
        if abs(default_parameters) < 10:
            uniform = 2 * random.random() - 1
            return uniform
        else:
            return initial_velocity(float(default_parameters))

def update_particle_position(position, velocity):
    # Recurse through the parameter data structure.
    if isinstance(position, dict):
        return {key: update_particle_position(value, velocity[key])
            for key, value in position.items()}
    elif isinstance(position, tuple):
        return tuple(update_particle_position(value, velocity[index])
            for index, value in enumerate(position))
    else:
        return position + velocity

def update_particle_velocity(postition, velocity, particle_best, global_best):
    # Recurse through the parameter data structure.
    if isinstance(postition, dict):
        return {key: update_particle_velocity(
                        postition[key], 
                        velocity[key], 
                        particle_best[key],
                        global_best[key])
            for key in postition.keys()}
    elif isinstance(postition, tuple):
        return tuple(update_particle_velocity(
                        postition[index], 
                        velocity[index], 
                        particle_best[index],
                        global_best[index])
            for index, value in enumerate(postition))
    else:
        # Update velocity.
        particle_bias = (particle_best - postition) * particle_strength * random.random()
        global_bias   = (global_best - postition)   * global_strength   * random.random()
        return velocity * velocity_strength + particle_bias + global_bias


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    assert(args.particles >= args.processes)

    if args.clear_scores:
        print("Removing Scores from Particle Swarm File %s."%swarm_path)
        swarm_data['best_score'] = None
        for entry in swarm_data:
            if isinstance(entry, int):
                swarm_data[entry]['best_score'] = None
        with open(swarm_path, 'w') as swarm_file:
            pprint.pprint(swarm_data, stream = swarm_file)
        sys.exit()

