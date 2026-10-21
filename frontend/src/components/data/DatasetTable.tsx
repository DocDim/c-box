import React from 'react';
import { DatasetPreview } from '../../types/api';

interface DatasetTableProps {
  dataset: DatasetPreview;
  selectedRowIndex: number | null;
  onSelectRow: (index: number, row: Record<string, any>) => void;
}

export const DatasetTable: React.FC<DatasetTableProps> = ({
  dataset,
  selectedRowIndex,
  onSelectRow
}) => {
  return (
    <div className="bg-slate-800/80 rounded-xl border border-slate-700 p-4 shadow-md">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-semibold text-white">
          Dataset Records: <span className="font-mono text-xs text-indigo-400">{dataset.filename}</span>
        </h3>
        <span className="text-xs text-slate-400">
          Showing first {dataset.preview_rows.length} of {dataset.total_rows} rows. Select a row to explain:
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300 border-collapse">
          <thead>
            <tr className="border-b border-slate-700 bg-slate-900/50 text-slate-400">
              <th className="p-2">Row #</th>
              {dataset.columns.map((col) => (
                <th key={col} className="p-2 font-mono whitespace-nowrap">{col}</th>
              ))}
              <th className="p-2 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {dataset.preview_rows.map((row, idx) => {
              const isSelected = selectedRowIndex === idx;
              return (
                <tr
                  key={idx}
                  onClick={() => onSelectRow(idx, row)}
                  className={`border-b border-slate-800/60 cursor-pointer transition ${
                    isSelected ? 'bg-indigo-950/60 text-white' : 'hover:bg-slate-700/40'
                  }`}
                >
                  <td className="p-2 font-mono text-slate-500 font-bold">{idx}</td>
                  {dataset.columns.map((col) => (
                    <td key={col} className="p-2 whitespace-nowrap font-mono">
                      {String(row[col])}
                    </td>
                  ))}
                  <td className="p-2 text-right">
                    <span
                      className={`px-2.5 py-1 rounded text-[11px] font-medium ${
                        isSelected
                          ? 'bg-indigo-600 text-white'
                          : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      {isSelected ? 'Active Row' : 'Explain'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};