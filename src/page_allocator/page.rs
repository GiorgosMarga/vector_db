use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    sync::Arc,
};

use ordered_float::NotNan;

use crate::{
    buffer_pool::page_guard::{NODE_SIZE, NODES_PER_PAGE, VECTOR_DIMS},
    errors::{DBError, Result},
    memgraph::node::{self, Node},
};

pub struct Page<T> {
    data: *mut u8,
    _marker: PhantomData<T>,
}

const PAGE_SIZE: usize = 64 * 1024;
const PAGE_ID_SIZE: usize = 8;
const VECTOR_CTR_SIZE: usize = 8;
const HEADER_SIZE: usize = 24;

pub struct VectorPage {}
pub struct DBHeaderPage {}
pub struct SegmentDirectoryPage {}
pub struct TranslationPage {}

impl<T> Page<T> {
    pub fn new(data: *mut u8) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    pub fn get_data(&self) -> &[u8; PAGE_SIZE] {
        unsafe { &*(self.data as *const [u8; PAGE_SIZE]) }
    }

    pub fn get_data_mut(&self) -> &mut [u8; PAGE_SIZE] {
        unsafe { &mut *(self.data as *mut [u8; PAGE_SIZE]) }
    }

    pub fn set_page_id(&mut self, page_id: u64) {
        let data: &mut [u8; PAGE_SIZE] = self.get_data_mut();
        data[0..8].copy_from_slice(&page_id.to_le_bytes());
    }
    pub fn get_page_id(&self) -> u64 {
        let data: &[u8; PAGE_SIZE] = self.get_data();
        u64::from_le_bytes(data[0..8].try_into().unwrap())
    }
}

//    [0..8)       node_id        : u64   (8 B)
//    [8..1208)    vector         : [f32; VECTOR_DIMS]  (1,200 B)
//    [1208..1210) neighbor_count : u16   (2 B)
//    [1210..1466) neighbor_ids   : [u64; MAX_NEIGHBORS] (256 B)
//    [1466..1536) padding        (70 B)
impl Page<VectorPage> {
    pub fn set_next_page_id(&mut self, next_page_id: u64) {
        let data: &mut [u8; PAGE_SIZE] = self.get_data_mut();
        data[8..16].copy_from_slice(&next_page_id.to_le_bytes());
    }
    pub fn get_next_page_id(&self) -> u64 {
        let data: &[u8; PAGE_SIZE] = self.get_data();
        u64::from_le_bytes(data[8..16].try_into().unwrap())
    }
    pub fn set_vector_ctr(&mut self, vector_ctr: u64) {
        let data: &mut [u8; PAGE_SIZE] = self.get_data_mut();
        data[16..24].copy_from_slice(&vector_ctr.to_le_bytes());
    }
    pub fn get_vector_ctr(&self) -> u64 {
        let data: &[u8; PAGE_SIZE] = self.get_data();
        u64::from_le_bytes(data[16..24].try_into().unwrap())
    }
    pub fn read_node(&self, slot: usize) -> Result<Node> {
        let vector_ctr = self.get_vector_ctr();
        if slot > vector_ctr as usize {
            return Err(DBError::Other("Read node: exceeded count"));
        }
        let data = self.get_data();
        let mut offset = HEADER_SIZE + slot * NODE_SIZE;
        let node_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let mut vec = [0f32; VECTOR_DIMS];
        for i in 0..VECTOR_DIMS {
            vec[i] = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }

        let count = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
        offset += 2;

        let neighbors = HashMap::new();
        let mut neighbors_at_0: BTreeSet<(NotNan<f32>, u64)> = BTreeSet::new();
        // neighbors.insert(0, );
        for _neighbor in 0..count {
            let neighbor_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            neighbors_at_0.insert((NotNan::new(f32::MAX).unwrap(), neighbor_id));
            offset += 8; // advance per neighbor
        }
        Ok(Node {
            id: node_id,
            vec: Arc::from(vec),
            neighbors,
        })
    }
    pub fn write_node(&mut self, slot: usize, node: &Node) -> Result<()> {
        let vector_ctr = self.get_vector_ctr();
        if slot > NODES_PER_PAGE {
            return Err(DBError::Other(
                "Vector page is full. Can't add more vectors.",
            ));
        }
        let data: &mut [u8; PAGE_SIZE] = self.get_data_mut();
        let mut offset = HEADER_SIZE + slot * NODE_SIZE;

        data[offset..offset + 8].copy_from_slice(&node.id.to_le_bytes());
        offset += 8;

        for val in node.vec.iter() {
            data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            offset += 4;
        }
        // save only neighbors from level 0
        if let Some(neighbors) = node.neighbors.get(&0) {
            let count = node.neighbors.len() as u16; // cast to u16                                                                        
            data[offset..offset + 2].copy_from_slice(&count.to_le_bytes());
            offset += 2; // advance past count                                                                
            for neighbor in neighbors {
                data[offset..offset + 8].copy_from_slice(&neighbor.1.to_le_bytes());
                offset += 8;
            }
        }
        self.set_vector_ctr(vector_ctr + 1);
        Ok(())
    }
    // fn get_vectors(&self) -> u64
}

// HEADER: 18 bytes -> ids = 2049
impl Page<TranslationPage> {
    pub fn set_next_translation_page(&mut self, next_translation_page: u64) {
        let data = self.get_data_mut();
        data[8..16].copy_from_slice(&next_translation_page.to_le_bytes());
    }
    pub fn get_next_translation_page(&self) -> u64 {
        let data = self.get_data();
        u64::from_le_bytes(data[8..16].try_into().unwrap())
    }
    pub fn set_ids_count(&mut self, ids_count: u16) {
        let data = self.get_data_mut();
        data[16..18].copy_from_slice(&ids_count.to_le_bytes());
    }
    pub fn get_ids_count(&self) -> u16 {
        let data = self.get_data();
        u16::from_le_bytes(data[16..18].try_into().unwrap())
    }
    pub fn get_nodes_map(&self) -> Vec<u64> {
        let mut map = Vec::new();
        let offset = 16;
        let data = self.get_data();
        let ids_count = self.get_ids_count();
        for i in 0..ids_count as usize {
            let id = u64::from_le_bytes(
                data[offset + (i * 8)..offset + (i * 8) + 8]
                    .try_into()
                    .unwrap(),
            );
            map.push(id);
        }
        map
    }
    pub fn set_nodes_map(&mut self, ids: &[u64]) {
        let offset = 16;
        let data = self.get_data_mut();
        for (i, &id) in ids.iter().enumerate() {
            data[offset + (i * 8)..offset + (i * 8) + 8].copy_from_slice(&id.to_le_bytes());
        }
    }
}

// 1248 bytes
#[derive(Debug)]
pub struct SegmentMetadata {
    pub segment_id: u64,
    pub vector_count: u64,
    pub first_data_page: u64,
    pub data_page_count: u32,
    pub first_translation_page: u64,
    pub translation_page_count: u32,
    pub entry_node_id: u64,
    pub centroid: [f32; 300],
}

// HEADER = 28 bytes -> 52 entries
impl Page<SegmentDirectoryPage> {
    pub fn set_next_page_id(&mut self, next_page_id: u64) {
        let data = self.get_data_mut();
        data[8..16].copy_from_slice(&next_page_id.to_le_bytes());
    }
    pub fn get_next_page_id(&self) -> u64 {
        let data = self.get_data();
        u64::from_le_bytes(data[8..16].try_into().unwrap())
    }
    pub fn set_entries_ctr(&mut self, entries_ctr: u32) {
        let data = self.get_data_mut();
        data[24..28].copy_from_slice(&entries_ctr.to_le_bytes());
    }
    pub fn get_entries_ctr(&self) -> u32 {
        let data = self.get_data();
        u32::from_le_bytes(data[24..28].try_into().unwrap())
    }

    pub fn append_entry(&mut self, metadata: SegmentMetadata) {
        let data = self.get_data_mut();
        let entries = self.get_entries_ctr();
        let mut offset = (28 + entries * 1248) as usize;
        data[offset..offset + 8].copy_from_slice(&metadata.segment_id.to_le_bytes());
        offset += 8;
        data[offset..offset + 8].copy_from_slice(&metadata.vector_count.to_le_bytes());
        offset += 8;
        data[offset..offset + 8].copy_from_slice(&metadata.first_data_page.to_le_bytes());
        offset += 8;
        data[offset..offset + 4].copy_from_slice(&metadata.data_page_count.to_le_bytes());
        offset += 4;
        data[offset..offset + 8].copy_from_slice(&metadata.first_translation_page.to_le_bytes());
        offset += 8;
        data[offset..offset + 4].copy_from_slice(&metadata.translation_page_count.to_le_bytes());
        offset += 4;
        data[offset..offset + 8].copy_from_slice(&metadata.entry_node_id.to_le_bytes());
        offset += 8;
        for val in metadata.centroid {
            data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            offset += 4;
        }
        self.set_entries_ctr(entries + 1);
    }

    pub fn get_entry(&self, segment_id: u64) -> Option<SegmentMetadata> {
        let data = self.get_data();
        let entries = self.get_entries_ctr();
        let mut offset = 28 as usize;
        for _ in 0..entries {
            let curr_segment_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            if curr_segment_id != segment_id {
                offset += 1248;
                continue;
            }
            offset += 8;
            let vector_count = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let first_data_page = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let data_page_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let first_translation_page =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let translation_page_count =
                u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let entry_node_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let mut centroid = [0f32; 300];
            for i in 0..300 {
                centroid[i] = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            return Some(SegmentMetadata {
                segment_id,
                vector_count,
                first_data_page,
                data_page_count,
                first_translation_page,
                translation_page_count,
                entry_node_id,
                centroid,
            });
        }
        None
    }
}

#[derive(Debug)]
pub struct DBHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub page_size: u32,
    pub segment_directory_page_id: u64,
    pub last_page_id: u64,
    pub next_vector_id: u64,
}
impl Page<DBHeaderPage> {
    pub fn set_header(
        &mut self,
        magic: [u8; 8],
        version: u32,
        page_size: u32,
        segment_directory_page_id: u64,
        last_page_id: u64,
        next_vector_id: u64,
    ) {
        let data = self.get_data_mut();
        data[8..16].copy_from_slice(&magic);
        println!("Set header as {:?}", magic);
        data[16..20].copy_from_slice(&version.to_le_bytes());
        data[20..24].copy_from_slice(&page_size.to_le_bytes());
        data[24..32].copy_from_slice(&segment_directory_page_id.to_le_bytes());
        data[32..40].copy_from_slice(&last_page_id.to_le_bytes());
        data[40..48].copy_from_slice(&next_vector_id.to_le_bytes());
    }

    pub fn get_header(&self) -> DBHeader {
        let data = self.get_data();
        let magic = data[8..16].try_into().unwrap();
        let version = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let page_size = u32::from_le_bytes(data[20..24].try_into().unwrap());
        let segment_directory_page_id = u64::from_le_bytes(data[24..32].try_into().unwrap());
        let last_page_id = u64::from_le_bytes(data[32..40].try_into().unwrap());
        let next_vector_id = u64::from_le_bytes(data[40..48].try_into().unwrap());

        DBHeader {
            magic,
            version,
            page_size,
            segment_directory_page_id,
            last_page_id,
            next_vector_id,
        }
    }

    pub fn get_segment_directory_page_id(&self) -> u64 {
        let data = self.get_data();
        u64::from_le_bytes(data[24..32].try_into().unwrap())
    }

    pub fn set_segment_directory_page_id(&mut self, segment_directory_page_id: u64) {
        let data = self.get_data_mut();
        data[24..32].copy_from_slice(&segment_directory_page_id.to_le_bytes());
    }

    pub fn set_last_page_id(&mut self, last_page_id: u64) {
        let data = self.get_data_mut();
        data[32..40].copy_from_slice(&last_page_id.to_le_bytes());
    }
}
// impl Page<SegmentMetadataPage> {
//     pub fn set_segment_id(&mut self, segment_id: u64) {
//         let data = self.get_data_mut();
//         data[8..16].copy_from_slice(&segment_id.to_le_bytes());
//     }
//     pub fn set_vector_count(&mut self, vector_count: u64) {
//         let data = self.get_data_mut();
//         data[16..24].copy_from_slice(&vector_count.to_le_bytes());
//     }
//     pub fn set_first_data_page_id(&mut self, page_id: u64) {
//         let data = self.get_data_mut();
//         data[24..32].copy_from_slice(&page_id.to_le_bytes());
//     }
//     pub fn set_data_page_count(&mut self, page_count: u64) {
//         let data = self.get_data_mut();
//         data[32..40].copy_from_slice(&page_count.to_le_bytes());
//     }
//     pub fn set_first_translation_page_id(&mut self, page_id: u64) {
//         let data = self.get_data_mut();
//         data[40..48].copy_from_slice(&page_id.to_le_bytes());
//     }
//     pub fn set_translation_page_count(&mut self, page_count: u64) {
//         let data = self.get_data_mut();
//         data[48..56].copy_from_slice(&page_count.to_le_bytes());
//     }
//     pub fn set_entry_node_id(&mut self, entry_node_id: u64) {
//         let data = self.get_data_mut();
//         data[56..64].copy_from_slice(&entry_node_id.to_le_bytes());
//     }
//     pub fn set_centroid(&mut self, centroid: &[f32; 300]) {
//         let data = self.get_data_mut();
//         for (i, val) in centroid.iter().enumerate() {
//             data[64 + (i * 4)..64 + (i * 4) + 4].copy_from_slice(&val.to_le_bytes());
//         }
//     }
// }
