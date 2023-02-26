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
const STATE: usize = 2;
const ACTION: usize = 4;

const AREA_WIDTH: f32 = 5.0;
const AREA_HEIGHT: f32 = 5.0;

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
struct Board {
    pos: [f32; 2],
    end: [f32; 2],
}

impl Board {
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();

        self.pos[0] = rng.gen_range(0..AREA_WIDTH as usize) as f32;
        self.pos[1] = rng.gen_range(0..AREA_HEIGHT as usize) as f32;
        self.end[0] = rng.gen_range(0..AREA_WIDTH as usize) as f32;
        self.end[1] = rng.gen_range(0..AREA_HEIGHT as usize) as f32;
    }
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
    board: Board,
    steps_since_last_merge: i32,
    survived_steps: i32,
    episode: i32,
    epsilon: f32,
    experience: Vec<Transition>,
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

        let mut board = Board::default();
        board.reset();

        Self {
            model: q_net.clone(),
            target: q_net.clone(),
            optimizer: sgd,
            board: board,
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

        // for _i_epoch in 0..2 {
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
    let mut last_time = Instant::now();

    while(!stop) {
        // if last_time.elapsed() > Duration::from_millis(100) {
            step(&mut model);
            last_time = Instant::now();
        // }
    }

    model.model.save("maze.npz");
}

fn clamp(input: f32, min: f32, max: f32) -> f32 {
    input.max(min).min(max)
}

fn step(model: &mut Model) {
    let observation = [
        model.board.pos[0] - model.board.end[0],
        model.board.pos[1] - model.board.end[1],  
    ];
    
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

    let prev_distance = ((model.board.end[0] - model.board.pos[0]).powi(2) + (model.board.end[1] - model.board.pos[1]).powi(2)).sqrt();

    match action {
        0 => {
            model.board.pos[0] -= 1.0;
        }
        1 => {
            model.board.pos[1] += 1.0;
        }
        2 => {
            model.board.pos[0] += 1.0;
        }
        3 => {
            model.board.pos[1] -= 1.0;
        }
        _ => {}
    }

    let mut hit = false;

    if(model.board.pos[0] < 0.0 || model.board.pos[0] > AREA_WIDTH ||
       model.board.pos[1] < 0.0 || model.board.pos[1] > AREA_HEIGHT) {
        hit = true;
        model.board.pos[0] = clamp(model.board.pos[0], 0.0, AREA_WIDTH); 
        model.board.pos[1] = clamp(model.board.pos[1], 0.0, AREA_HEIGHT);
    }

    

    // println!("Pos: {:?}, End: {:?}, Dir: {:?}", model.board.pos, model.board.end, action);

    let next_distance = ((model.board.end[0] - model.board.pos[0]).powi(2) + (model.board.end[1] - model.board.pos[1]).powi(2)).sqrt();

    let next_observation = [
        model.board.pos[0] - model.board.end[0],
        model.board.pos[1] - model.board.end[1]
    ];

    let reward = (prev_distance - next_distance) / 10.0;
    let reached = (model.board.pos[0] == model.board.end[0] && model.board.pos[1] == model.board.end[1]);

    if(reached) {
        let r = if !reached { (model.survived_steps as f32 * -0.04) } else { 2.0 - (model.survived_steps as f32 * 0.04) };
        // let r = if !reached { -next_distance } else { 100.0 - model.survived_steps as f32 };
        
        println!("Survived: {:?}, Reward: {:?}, Pos: {:?}, End: {:?}", model.survived_steps, r, model.board.pos, model.board.end);
        model.push_experience((observation, action, 1.0, next_observation, 1.0));

        model.board.reset();

        model.survived_steps = 0;
        model.episode += 1;
    } else {
        model.survived_steps += 1;
        model.push_experience((observation, action, if hit { -0.1 } else { -0.04 }, next_observation, 0.0));
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
