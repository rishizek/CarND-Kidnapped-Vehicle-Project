/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  default_random_engine gen;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  double sample_x, sample_y, sample_theta;
  double weight = 1./num_particles;
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    while (sample_theta >  M_PI) sample_theta -= 2.*M_PI;
    while (sample_theta < -M_PI) sample_theta += 2.*M_PI;

    particle.id = i;
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
    particle.weight = weight;
    particles.push_back(particle);
    weights.push_back(weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  //extract values for better readability
  
  double x, y, yaw;
  default_random_engine gen;
  //add noise
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  for (int i = 0; i < num_particles; ++i) {
    x = particles[i].x;
    y = particles[i].y;
    yaw = particles[i].theta;
    //avoid division by zero
    if (fabs(yaw_rate) > 0.001) {
      x += velocity/yaw_rate * (sin(yaw + yaw_rate*delta_t) - sin(yaw)) + dist_x(gen);
      y += velocity/yaw_rate * (cos(yaw) - cos(yaw+yaw_rate*delta_t)) + dist_y(gen);
    } else {
      x += velocity*delta_t*cos(yaw) + dist_x(gen);
      y += velocity*delta_t*sin(yaw) + dist_y(gen);
    }
    
    yaw += yaw_rate*delta_t + dist_theta(gen);
    
    //write predicted position to each particle
    particles[i].x = x;
    particles[i].y = y;
    particles[i].theta = yaw;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  double x_o, y_o, x_p, y_p, dist_pred;
  for (int i = 0; i < observations.size(); ++i) {
    double min_dist = std::numeric_limits<double>::max();
    x_o = observations[i].x;
    y_o = observations[i].y;
    for (int j = 0; j < predicted.size(); ++j) {
      x_p = predicted[j].x;
      y_p = predicted[j].y;
      dist_pred = dist(x_o, y_o, x_p, y_p);
      if (dist_pred < min_dist) {
        observations[i].id = predicted[j].id;
        min_dist = dist_pred;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double x_p, y_p, yaw_p;
  for (int i = 0; i < num_particles; ++i) {
    x_p = particles[i].x;
    y_p = particles[i].y;
    yaw_p = particles[i].theta;
    
    double x_o, y_o;
    double x_m, y_m;
    vector<LandmarkObs> transformedObs;
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs tObs;
      x_o = observations[j].x;
      y_o = observations[j].y;
      //transform each observation marker from the particle's coordinates to the map's coordinates
      x_m = x_p + x_o * cos(yaw_p) - y_o * sin(yaw_p);
      y_m = y_p + x_o * sin(yaw_p) + y_o * cos(yaw_p);
      tObs.x = x_m;
      tObs.y = y_m;
      tObs.id = observations[j].id;
      transformedObs.push_back(tObs);
    }

    //select landmarks that are in sensor_range
    vector<LandmarkObs> candidate_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      int id_i = map_landmarks.landmark_list[j].id_i;
      float x_f = map_landmarks.landmark_list[j].x_f;
      float y_f = map_landmarks.landmark_list[j].y_f;
      if (dist(x_p, y_p, x_f, y_f) <= sensor_range) {
        LandmarkObs landmark;
        landmark.id = id_i;
        landmark.x = x_f;
        landmark.y = y_f;
        candidate_landmarks.push_back(landmark);
      }
    }
    
    dataAssociation(candidate_landmarks, transformedObs);
    // update weights
    // The particles final weight will be calculated as the product of each measurement's Multivariate-Gaussian probability.
    double prob;
    double log_p = 0.0;
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for (int j = 0; j < transformedObs.size(); ++j) {
      LandmarkObs obs = transformedObs[j];
      double x = obs.x;
      double y = obs.y;
      int id_j = obs.id;
      associations.push_back(id_j);
      sense_x.push_back(x);
      sense_y.push_back(y);
      
      bool found_landmark = false;
      LandmarkObs landmark;
      for (int k = 0; k < candidate_landmarks.size(); ++k) {
        if (candidate_landmarks[k].id == id_j) {
          landmark = candidate_landmarks[k];
          found_landmark = true;
          break;
        }
      }
      
      if (!found_landmark) {
        log_p = std::numeric_limits<double>::min();
        break;
      }
      
      double mu_x = landmark.x;
      double mu_y = landmark.y;
      double diff_x = x-mu_x;
      double diff_y = y-mu_y;
      log_p += -(diff_x*diff_x/(2.*sig_x*sig_x) + diff_y*diff_y/(2.*sig_y*sig_y))
               - log(2.*M_PI*sig_x*sig_y);
    }
    prob = exp(log_p);
    particles[i].weight = prob;
    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
  }

  for (int i = 0; i < num_particles; ++i) {
    weights[i] = particles[i].weight; // no need for normalization
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<> dist_particles(weights.begin(), weights.end());
  vector<Particle> particles_tmp;
  for (int i = 0; i < num_particles; ++i) {
    int idx = dist_particles(gen);
    particles_tmp.push_back(particles[idx]);
    weights[i] = particles[idx].id;
  }
  particles = particles_tmp;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
