"""CPU unit tests for the independent audio adaptation policy/objective."""
import unittest
import torch
from train_depth_audio import audio_parameter
from jmoshivis.loss import compute_loss_with_mask

class DepthAudioTests(unittest.TestCase):
    def test_only_audio_parameters(self):
        for name in ['depformer.layers.0.self_attn.in_proj_weight','depformer_in.0.weight',
                     'depformer_emb.0.weight','depformer_text_emb.weight','audio_linears.7.weight']:
            self.assertTrue(audio_parameter(name),name)
        for name in ['llm.transformer.layers.0.cross_attention.gate.alpha.weight',
                     'llm.text_linear.weight','llm.emb.0.weight','image_prefix.proj.weight']:
            self.assertFalse(audio_parameter(name),name)

    def test_every_codebook_receives_gradient_and_padding_is_excluded(self):
        logits=torch.randn(1,8,4,16,requires_grad=True)
        target=torch.randint(0,16,(1,8,4))
        mask=torch.ones_like(target,dtype=torch.bool);mask[:,:,-1]=False
        with torch.no_grad():logits[:,:,-1]=float('nan')
        losses=[compute_loss_with_mask(logits[:,i:i+1],target[:,i:i+1],mask[:,i:i+1],mode='audio') for i in range(8)]
        torch.stack(losses).mean().backward()
        for i in range(8):
            self.assertTrue(torch.isfinite(logits.grad[:,i]).all())
            self.assertGreater(float(logits.grad[:,i,:-1].abs().sum()),0)
            self.assertEqual(float(logits.grad[:,i,-1].abs().sum()),0)

if __name__=='__main__':unittest.main()
