use std::collections::{BTreeSet, HashMap, HashSet};

use ordered_float::NotNan;

use crate::memgraph::node;

#[derive(Debug, Clone)]
pub struct Node {
    pub id: u64,
    pub neighbors: HashMap<u64, BTreeSet<(NotNan<f32>, u64)>>,
}
impl Node {
    pub fn add_neighbor(
        &mut self,
        node_id: u64,
        level: u64,
        dist: NotNan<f32>,
        m: usize,
    ) -> (bool, Option<u64>) {
        let neighbors = self.neighbors.entry(level).or_default();
        if neighbors.len() < m {
            neighbors.insert((dist, node_id));
            return (true, None);
        }
        if neighbors.last().unwrap().0 > dist {
            let evicted_id = neighbors.pop_last().unwrap().1;
            neighbors.insert((dist, node_id));
            return (true, Some(evicted_id));
        }

        (false, None)
    }

    pub fn delete_neighbor(&mut self, node_id: u64, level: u64) {
        let neighbors = self.neighbors.get_mut(&level).unwrap();
        neighbors.retain(|v| v.1 != node_id);
    }
}
