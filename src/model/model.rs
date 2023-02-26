#![allow(unused)]
use dfdx::{
    losses::mse_loss,
    optim::{Momentum, Sgd, SgdConfig, Adam, AdamConfig, RMSprop, RMSpropConfig},
    prelude::{*, modules::Linear}, nn, gradients::Tape,
};
use rand::prelude::*;
use std::{
    sync::Arc,
    time::{Duration, Instant}, borrow::BorrowMut,
};

// #[cfg(not(feature = "cuda"))]
// type Dev = dfdx::tensor::Cpu;

// #[cfg(feature = "cuda")]
type Dev = dfdx::tensor::Cuda;

type Mlp = (
    (Linear<STATE, 128, f32, Dev>, ReLU),
    (Linear<128, 64, f32, Dev>, ReLU),
    Linear<64, ACTION, f32, Dev>
);

const NEXT_STATE_DISCOUNT: f32 = 0.9;
const EPSILON_DECAY: f32 = 0.001;

const BATCH: usize = 64;
const STATE: usize = 2;
const ACTION: usize = 4;

type Observation = [f32; STATE];
type Action = [f32; ACTION];

type Transition = (Observation, usize, f32, Observation, f32);
type TransitionTuple = (
    Tensor<Rank2<BATCH, STATE>, f32, Dev>,
    Tensor<Rank1<BATCH>, usize, Dev>,
    Tensor<Rank1<BATCH>, f32, Dev>,
    Tensor<Rank2<BATCH, STATE>, f32, Dev>,
    Tensor<Rank1<BATCH>, f32, Dev>,
);

pub struct Model {
    pub dev: Dev,
    pub dqn: Mlp,
    pub target: Mlp,
    pub optimizer: Sgd<Mlp, f32>,
    pub steps_since_last_merge: i32,
    pub survived_steps: i32,
    pub episode: i32,
    pub epsilon: f32,
    pub experience: Vec<Transition>,
    pub prev_observation: Observation
}

impl Model {
    pub fn new() -> Self {
        let dev: Dev = Default::default();
        // println!("Device: {:?}", dev);
        let mut dqn: Mlp = nn::BuildModule::build(&dev);

        let mut rng = rand::thread_rng();
        dqn.reset_params();

        let mut sgd = Sgd::new(
            &dqn,
            SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        });

        let mut adam = Adam::new(
            &dqn,
            AdamConfig {
            lr: 1e-3,
            ..Default::default()
        });

        let mut rms = RMSprop::new(
            &dqn,
            RMSpropConfig {
                lr: 0.00025,
                alpha: 0.95,
                eps: 0.01,
                ..Default::default()
            }
        );

        Self {
            dev: dev,
            dqn: dqn.clone(),
            target: dqn.clone(),
            optimizer: sgd,
            steps_since_last_merge: 0,
            survived_steps: 0,
            episode: 0,
            epsilon: 1.0,
            experience: Vec::new(),
            prev_observation: Observation::default()
        }
    }

    pub fn push_experience(&mut self, experience: Transition) {
        self.experience.push(experience);

        if self.experience.len() > 10000 {
            self.experience = self.experience[5000..].to_vec();
        }
    }

    pub fn get_batch_tensors(&self, sample_indexes: [usize; BATCH]) -> TransitionTuple {
        let batch: [Transition; BATCH] = sample_indexes.map(|i| self.experience[i]);
        let mut states = [[0.0; STATE]; BATCH];
        let mut actions = [0; BATCH];
        let mut rewards = [0.0; BATCH];
        let mut next_states = [[0.0; STATE]; BATCH];
        let mut done = [0.0; BATCH];

        for (i, (s, a, r, s_n, d)) in batch.iter().enumerate() {
            states[i] = *s;
            actions[i] = 1 * a;
            rewards[i] = *r;
            next_states[i] = *s_n;
            done[i] = *d;
        }

        (
            self.dev.tensor(states), 
            self.dev.tensor(actions), 
            self.dev.tensor(rewards), 
            self.dev.tensor(next_states), 
            self.dev.tensor(done)
        )
    }

    pub fn train(&mut self) {
        let mut rng = rand::thread_rng();

        let distribution = rand::distributions::Uniform::from(0..self.experience.len());

        let indices = [(); BATCH].map(|i| distribution.sample(&mut rng));
        let (s, a, r, sn, done) = self.get_batch_tensors(indices);

        // println!(
        //     "I {:?}\n
        //      S {:?}\n
        //      A {:?}\n
        //      R {:?}\n
        //     SN {:?}\n
        //      D {:?}",
        //     indices, s.array(), a.array(), r.array(), sn.array(), done.array()
        // );

        // println!("Total Reward: {}", r.clone().sum().array());

        for _epoch in 0..1 {
            let mut total_loss = 0.0;

            let q_values = self.dqn.forward(s.trace());
            let action_qs = q_values.select(a.clone());

            let next_q_values = self.target.forward(sn.clone());
            let max_next_q = next_q_values.max::<Rank1<BATCH>, _>();

            let target_q = (max_next_q * (-done.clone() + 1.0)) * 0.99 + r.clone();
            
            let loss = huber_loss(action_qs, target_q, 1.);
            total_loss += loss.array();

            let gradients = loss.backward();
            self.optimizer.update(&mut self.dqn, gradients).expect("Unused params");
            // self.target.clone_from(&self.dqn);

            // println!(
            //     "Epoch {}: loss={:#.3}",
            //     _epoch,
            //     total_loss
            // );
        }

    }

    pub fn forward(&mut self, observation: Observation) -> usize {
        let mut rng = rand::thread_rng();
        let rando = rng.gen_range(0.0..1.0);
        
        let action = match self.epsilon > rando {
            true => {
                rng.gen_range(0..ACTION) as usize
            }
            false => {
                let tensor_observation = self.dev.tensor(observation);
                let prediction = self.dqn.forward(tensor_observation);
                prediction
                    .array()
                    .iter()
                    .enumerate()
                    .max_by(|(_, value0), (_, value1)| value0.total_cmp(value1))
                    .map(|(idx, _)| idx)
                    .unwrap()
            }
        };

        action
    }
}
