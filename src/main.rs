#![allow(unused)]
use std::{
    env,
    fs,
    io::Error as IoError,
    net::{SocketAddr, TcpListener, TcpStream},
    time::{Duration, self, Instant}, sync::Arc, thread::{spawn, self}, collections::{HashMap, HashSet}
};
use rand::seq::SliceRandom;
use tungstenite::{Message, Result, WebSocket, accept};

mod model;
use model::model::Model;

struct Stats {
    total: u32,
    success: u32,
}

fn main () {
    let server = TcpListener::bind("127.0.0.1:8080").unwrap();
    println!("Listening on: {}", server.local_addr().unwrap());
    let mut steps = 0;

    for stream in server.incoming() {
        spawn(move || {
            let a = stream.unwrap();
            println!("Incoming connection: {}", a.peer_addr().unwrap());
            let mut websocket = accept(a).unwrap();
            let mut model = Model::new();
            let mut stats = Stats {
                success: 0,
                total: 0
            };

            loop {
                let msg = websocket.read_message().unwrap();

                // We do not want to send back ping/pong messages.
                if msg.is_binary() || msg.is_text() {
                    step(&mut model, msg, &mut websocket, &mut stats);

                    if model.episode > 500 {
                        let ten_millis = time::Duration::from_millis(200);
                        let now = time::Instant::now();

                        thread::sleep(ten_millis);
                    }
                }
            }
        });
    }
}

fn step(model: &mut Model, msg: Message, socket: &mut WebSocket<TcpStream>, stats: &mut Stats) {
    let slice = msg.into_data();
    let mut stage = f32::from_le_bytes(slice[0..4].try_into().unwrap());
    let mut x = f32::from_le_bytes(slice[4..8].try_into().unwrap());
    let mut y = f32::from_le_bytes(slice[8..12].try_into().unwrap());
    let distance = ((x - 3.0).powi(2) + (y - 2.0).powi(2)).sqrt();

    let observation = [x, y];

    if stage == 0.0 {
        let action = model.forward(observation);

        model.epsilon = (model.epsilon - 0.0002).max(0.05);

        let mut send = ""; 

        match action {
            0 => {
                send = "N";
            }
            1 => {
                send = "E";
            }
            2 => {
                send = "S";
            }
            3 => {
                send = "W";
            }
            _ => {}
        }

        model.prev_observation = observation;

        socket.write_message(Message::Text(send.to_owned())).unwrap();
    } else {
        let action = f32::from_le_bytes(slice[12..16].try_into().unwrap()) as usize;
        
        let reached_target = x == 3.0 && y == 2.0;
        let hit_bad = x == 3.0 && y == 1.0;
    
        if(reached_target || hit_bad || model.survived_steps > 49) {
            println!("Survived: {:?}, Pos: {:?}", model.survived_steps, observation);
            model.push_experience((model.prev_observation, action, if reached_target { stats.success += 1; 1.0 } else { -1.0 }, observation, 1.0));
            stats.total += 1;
            println!("Reached Target: {}. Accuracy: {:#.3}", reached_target, stats.success as f32 / stats.total as f32);

            model.survived_steps = 0;
            model.episode += 1;
            socket.write_message(Message::Text("R".to_owned())).unwrap();
        } else {
            model.survived_steps += 1;
            model.push_experience((model.prev_observation, action, -0.04, observation, 0.0));
            socket.write_message(Message::Text("NEXT".to_owned())).unwrap();
        }

        if model.experience.len() > 99 {
            model.train();
        }
    
        if model.steps_since_last_merge > 10 {
            model.target = model.dqn.clone();
            model.steps_since_last_merge = 0;
        } else {
            model.steps_since_last_merge += 1;
        }
    }
}