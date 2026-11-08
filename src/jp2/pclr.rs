//! The palette box (pclr), defined in I.5.3.4.

use alloc::vec::Vec;

use crate::error::{FormatError, Result, ValidationError, bail};
use crate::j2c::BuildBudget;
use crate::jp2::ImageBoxes;
use crate::reader::BitReader;

pub(crate) fn parse(boxes: &mut ImageBoxes, data: &[u8], budget: &mut BuildBudget<'_>) -> Result<()> {
    let mut reader = BitReader::new(data);
    let num_entries = reader.read_u16().ok_or(FormatError::InvalidBox)? as usize;
    let num_components = reader.read_byte().ok_or(FormatError::InvalidBox)? as usize;

    if num_entries == 0 || num_components == 0 {
        bail!(FormatError::InvalidBox);
    }

    let mut columns = Vec::new();
    budget.reserve_exact(&mut columns, num_components)?;
    for _ in 0..num_components {
        let descriptor = reader.read_byte().ok_or(FormatError::InvalidBox)?;
        let bit_depth = (descriptor & 0x7F)
            .checked_add(1)
            .ok_or(FormatError::InvalidBox)?;
        let is_signed = (descriptor & 0x80) != 0;

        if is_signed {
            bail!(FormatError::InvalidBox);
        }

        columns.push(PaletteColumn { bit_depth });
    }

    let mut entries = Vec::new();
    budget.reserve_exact(&mut entries, num_entries)?;

    for _ in 0..num_entries {
        let mut row = Vec::new();
        budget.reserve_exact(&mut row, num_components)?;

        for column in &columns {
            let num_bytes = (column.bit_depth as usize).div_ceil(8).max(1);
            let raw_bytes = reader
                .read_bytes(num_bytes)
                .ok_or(FormatError::InvalidBox)?;
            let mut raw_value = 0_u64;
            for &byte in raw_bytes {
                raw_value = (raw_value << 8) | byte as u64;
            }

            row.push(raw_value);
        }

        entries.push(row);
    }

    if let Some(old) = boxes.palette.replace(PaletteBox { entries, columns }) {
        let cells = old.entries.len().checked_mul(old.columns.len())
            .ok_or(ValidationError::MemoryLimitExceeded)?;
        budget.release_bytes(BuildBudget::bytes::<u64>(cells)?)?;
        budget.release_bytes(BuildBudget::bytes::<Vec<u64>>(old.entries.len())?)?;
        budget.release_bytes(BuildBudget::bytes::<PaletteColumn>(old.columns.len())?)?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct PaletteBox {
    pub(crate) entries: Vec<Vec<u64>>,
    pub(crate) columns: Vec<PaletteColumn>,
}

impl PaletteBox {
    #[inline(always)]
    pub(crate) fn map(&self, entry: usize, column: usize) -> Option<u64> {
        self.entries
            .get(entry)
            .and_then(|row| row.get(column))
            .copied()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PaletteColumn {
    pub(crate) bit_depth: u8,
}
