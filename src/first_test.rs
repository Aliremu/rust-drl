use dfdx::{
    losses::mse_loss,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};
use rand::prelude::*;
use std::time::{Instant, Duration};

const NEXT_STATE_DISCOUNT: f32 = 0.9;
const EPSILON_DECAY: f32 = 0.0002;

const BATCH: usize = 64;
const STATE: usize = 1;
const ACTION: usize = 2;

type Observation = [f32; STATE];
type Action = [f32; ACTION];

type Transition = (Observation, usize, f32, Observation, f32);
type TransitionTuple = (
    Tensor2D<BATCH, STATE>,
    [usize; BATCH],
    Tensor1D<BATCH>,
    Tensor2D<BATCH, STATE>,
    Tensor1D<BATCH>
);

#[derive(Default, Debug)]
struct Vec2 {
    x: f32,
    y: f32,
}
#[derive(Default, Debug)]
struct Car {
    pos: Vec2,
}

type QNetwork = (
    (Linear<STATE, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    Linear<32, ACTION>,
);

#[derive(Debug)]
struct Model {
    model: QNetwork,
    target: QNetwork,
    optimizer: Sgd<QNetwork>,
    car: Car,
    steps_since_last_merge: i32,
    survived_steps: i32,
    episode: i32,
    epsilon: f32,
    experience: Vec<Transition>,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x: x, y: y }
    }
}

impl Model {
    pub fn new() -> Self {
        let mut q_net = QNetwork::default();

        let mut rng = rand::thread_rng();
        q_net.reset_params(&mut rng);

        let mut sgd = Sgd::new(SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        });

        Self {
            model: q_net.clone(),
            target: q_net.clone(),
            optimizer: sgd,
            car: Car::default(),
            steps_since_last_merge: 0,
            survived_steps: 0,
            episode: 0,
            epsilon: 1.0,
            experience: Vec::new(),
        }
    }

    pub fn push_experience(&mut self, experience: Transition) {
        self.experience.push(experience);
    }

    pub fn get_batch_tensors(&self, sample_indexes: [usize; BATCH]) -> TransitionTuple {
        let batch: [Transition; BATCH] = sample_indexes.map(|i| self.experience[i]);
        let mut states: Tensor2D<BATCH, STATE> = Tensor2D::zeros();
        let mut actions: [usize; BATCH] = [0; BATCH];
        let mut rewards: Tensor1D<BATCH> = Tensor1D::zeros();
        let mut next_states: Tensor2D<BATCH, STATE> = Tensor2D::zeros();
        let mut done: Tensor1D<BATCH> = Tensor1D::zeros();
        for (i, (s, a, r, s_n, d)) in batch.iter().enumerate() {
            states.mut_data()[i] = *s;
            actions[i] = 1 * a;
            rewards.mut_data()[i] = *r;
            next_states.mut_data()[i] = *s_n;
            done.mut_data()[i] = *d;
        }
        (states, actions, rewards, next_states, done)
    }

    pub fn train(&mut self) {
        let mut rng = rand::thread_rng();

        // let batch_indexes = [(); BATCH].map(|_| rng.gen_range(0..self.experience.len()));
        
        let distribution = rand::distributions::Uniform::from(0..self.experience.len());

        let indices = [(); BATCH].map(|i| distribution.sample(&mut rng));
        let (s, a, r, sn, done) = self.get_batch_tensors(indices);

        // for _i_epoch in 0..15 {
            let next_q_values: Tensor2D<BATCH, ACTION> = self.model.forward(sn.clone());
            let max_next_q: Tensor1D<BATCH> = next_q_values.max();
            let target_q = 0.99 * mul(max_next_q, 1.0 - done.clone()) + r.clone();
            // forward through model, computing gradients
            let q_values: Tensor2D<BATCH, ACTION, OwnedTape> = self.model.forward(s.trace());
            // println!("{:?}", a);
            let action_qs: Tensor1D<BATCH, OwnedTape> = q_values.select(&a);

            let loss = huber_loss(action_qs, target_q, 1.);
            let loss_v = *loss.data();

            // run backprop
            let gradients = loss.backward();
            self.optimizer
                .update(&mut self.model, gradients)
                .expect("Unused params");
            // println!("{:.2} ", loss_v);
        // }
    }
}

fn main() {
    let mut model = Model::new();
    let mut stop = false;
    let mut lastTime = Instant::now();

    while(!stop) {
        if(lastTime.elapsed() > Duration::from_millis(1)) {
            step(&mut model);
            lastTime = Instant::now();
        }
    }
}

fn step(model: &mut Model) {
    let observation = [model.car.pos.x];
    let mut rng = rand::thread_rng();

    let action = match model.epsilon > rand::random::<f32>() {
        true => {
            rng.gen_range(0..ACTION) as usize
        },
        false => {
            let tensor_observation: Tensor1D<STATE> = TensorCreator::new(observation);
            let prediction: Tensor1D<ACTION> = model.model.forward(tensor_observation);
            prediction.data().iter().enumerate().max_by(|(_, value0), (_, value1)| value0.total_cmp(value1)).map(|(idx, _)| idx).unwrap()
        }
    };

    model.epsilon = (model.epsilon - EPSILON_DECAY).max(0.05);

    match action {
        1 => {
            model.car.pos.x += 1.0;
        }
        0 => {
            model.car.pos.x -= 1.0;
        }
        _ => {}
    }

    let next_observation = [model.car.pos.x];

    if(model.car.pos.x > 100.0 || model.car.pos.x < -100.0 || model.survived_steps > 499) {
        println!("Survived: {:?}, Reward: {:?}", model.survived_steps, model.car.pos.x);
        model.push_experience((observation, action, model.car.pos.x - model.survived_steps as f32, next_observation, 1.0));

        model.car.pos.x = 0.0;
        model.survived_steps = 0;
        model.episode += 1;
    } else {
        model.survived_steps += 1;
        model.push_experience((observation, action, model.car.pos.x - model.survived_steps as f32, next_observation, 0.0));
    }

    if model.experience.len() > BATCH {
        model.train();
    }

    if model.steps_since_last_merge > 10 {
        model.target = model.model.clone();
        model.steps_since_last_merge = 0;
    } else {
        model.steps_since_last_merge += 1;
    }
}
