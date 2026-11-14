from io import BytesIO
import unittest

import torch
import input_batching as batching


class InputBatchingTests(unittest.TestCase):
    def test_finite_window_stable_sort_and_identity_coverage(self):
        lengths=[12]*15+[18]+[10]*16
        arrival=list(reversed(range(32)))
        groups=list(batching.window_groups(lengths,arrival,32))
        self.assertEqual(groups[0][0],arrival)
        flattened=[i for _,batches in groups for group in batches for i in group]
        self.assertEqual(sorted(flattened),list(range(32)))
        self.assertEqual([i for i in flattened if lengths[i]==10],arrival[:16])
        for window in (0,17,8):
            with self.assertRaises(ValueError):list(batching.window_groups(lengths,arrival,window))
        with self.assertRaises(ValueError):list(batching.window_groups(lengths,[0]*32,16))

    def test_actual_tensor_accounting_and_standalone_frames(self):
        inputs=[torch.full((n,192),i+1,dtype=torch.bfloat16) for i,n in enumerate((12,18,7))]
        packet=batching.make_batch(inputs,[8,4,2],torch)
        cost=batching.batch_bytes(*packet[:3])
        self.assertEqual(cost,{'input_bytes':3*18*384,'length_bytes':12,'mask_bytes':54,'allocated_positions':54})
        stream=BytesIO();batching.write_batch(stream,packet);data=stream.getvalue();stream.seek(0)
        restored=batching.read_batch(stream,torch)
        for (identity,tensor),expected in zip(batching.unpack_batch(*restored,torch),inputs,strict=True):
            self.assertIn(identity,(8,4,2));self.assertTrue(torch.equal(tensor,expected))
        self.assertIsNone(batching.read_batch(stream,torch))
        for bad in (data[:5],data[:-1]):
            with self.assertRaises(ValueError):batching.read_batch(BytesIO(bad),torch)

    def test_rejects_bad_mask_padding_lengths_and_duplicates(self):
        tensor=torch.ones((2,192),dtype=torch.bfloat16)
        packet=batching.make_batch([tensor,tensor[:1]],[0,1],torch)
        for field in ('mask','length','padding'):
            batch,lengths,mask,ids=packet
            batch,lengths,mask=batch.clone(),lengths.clone(),mask.clone()
            if field=='mask':mask[1,1]=True
            elif field=='length':lengths[1]=3
            else:batch[1,1,0]=1
            with self.assertRaises(ValueError):batching.unpack_batch(batch,lengths,mask,ids,torch)
        with self.assertRaises(ValueError):batching.make_batch([tensor,tensor],[0,0],torch)

    def test_equal_lengths_and_sorting_within_batch_cannot_save_positions(self):
        lengths=[12]*40;arrival=list(range(40))
        totals=[]
        for window in (16,64,128):
            total=sum(len(group)*max(lengths[i] for i in group) for _,groups in batching.window_groups(lengths,arrival,window) for group in groups)
            totals.append(total)
        self.assertEqual(totals,[480]*3)
        varied=list(range(1,19))*2
        def allocated(sort):
            return sum(len(g)*max(varied[i] for i in g) for _,groups in batching.window_groups(varied,list(range(36)),16,sort=sort) for g in groups)
        self.assertEqual(allocated(False),allocated(True))


if __name__=='__main__':unittest.main()
