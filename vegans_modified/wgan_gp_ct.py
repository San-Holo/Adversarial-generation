import torch
from .gan import GAN


class WGANGP_CT(GAN):
    """
    Wasserstein GAN with gradient penalty
    https://arxiv.org/abs/1704.00028
    """

    def train(self, critic_iters=5, long_critic_iters=100, lambda_1=10, lambda_2=10, M=0):
        """
        :param critic_iters:
        :param long_critic_iters:
        :param lambda_gp:
        :param M:
        :return:
        """

        def _grad_penalty(real, fake):
            """
            Computes the gradient penalty.
            The gradient is taken for linear interpolations between real and fake samples.
            """
            assert real.size() == fake.size(), 'real and fake mini batches must have same size'
            batch_size = real.size(0)
            epsilon = torch.rand(batch_size, *[1 for _ in range(real.dim()-1)], device=self.device)
            x_hat = (epsilon * real + (1. - epsilon) * fake).requires_grad_(True)
            output = self.discriminator(x_hat)
            grads = torch.autograd.grad(
                outputs=output,
                inputs=x_hat,
                grad_outputs=torch.ones(output.size(), device=self.device),
                create_graph=True,
				retain_graph=True
            )[0]
            return ((grads.norm(2, dim=1) - 1) ** 2).mean()

        def _consistency_term(real, discriminator):
            """
            Computes the consistency term.
            Max is not needed in the implementation since l2 norm is always non-negative.
            """
            output_first, output_pre_first = discriminator(real, with_pre_last=True)
            output_second, output_pre_second = discriminator(real, with_pre_last=True)
            disc_diff = torch.norm((output_first - output_second), p=2, dim=1)
            disc_diff_pre = torch.norm((output_pre_first - output_pre_second), p=2, dim=1)
            return (disc_diff + 0.1 * disc_diff_pre - M).mean()

        gen_iters = 0  # the generator is not trained every iteration
        for epoch in range(self.nr_epochs):
            for minibatch_iter, data in enumerate(self.dataloader):

                # the number of mini batches we'll train the critic before training the generator
                if gen_iters < 25 or gen_iters % 500 == 0:
                    D_iters = long_critic_iters
                else:
                    D_iters = critic_iters

                real = data.to(self.device)
                batch_size = real.size(0)

                """ Train the critic
                """
                self.optimizer_D.zero_grad()
                noise = torch.randn(batch_size, self.nz, device=self.device)
                fake = self.generator(noise).detach()
                # Sign is inverse of paper because in paper it's a maximization problem
                fake_d = self.discriminator(fake)
                real_d = self.discriminator(real)
                loss_D = fake_d.mean() - real_d.mean()
                gp = _grad_penalty(real, fake)
                ct = _consistency_term(real,self.discriminator)
                loss_D_with_two_penalties = loss_D + lambda_1 * gp + lambda_2 * ct

                loss_D_with_two_penalties.backward()
                self.optimizer_D.step()

                loss_G = None
                if self.global_iter % D_iters == 0:
                    """ Train the generator every [Diters]
                    """
                    self.optimizer_G.zero_grad()
                    fake = self.generator(noise)
                    fake_d = self.discriminator(fake)
                    loss_G = -torch.mean(fake_d)
                    loss_G.backward()
                    self.optimizer_G.step()
                    gen_iters += 1

                # End iteration
                self._end_iteration(epoch, minibatch_iter, loss_G.item() if loss_G is not None else None, loss_D.item(), GP=gp, CT=ct)

        return self.samples, self.D_losses, self.G_losses
