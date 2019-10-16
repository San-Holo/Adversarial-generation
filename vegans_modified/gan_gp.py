import torch
import torch.nn as nn
from .gan import GAN


class GANGP(GAN):
    """
    NS GAN with gradient penaltys
    """
    def __init__(self,
                 generator,
                 discriminator,
                 dataloader,
                 optimizer_D=None,
                 optimizer_G=None,
                 nz=100,
                 device='cpu',
                 ngpu=0,
                 fixed_noise_size=64,
                 nr_epochs=5,
                 save_every=500,
                 print_every=50,
                 init_weights=False):
        super().__init__(generator,
                 discriminator,
                 dataloader,
                 optimizer_D,
                 optimizer_G,
                 nz,
                 device,
                 ngpu,
                 fixed_noise_size,
                 nr_epochs,
                 save_every,
                 print_every,
                 init_weights)
        self.BCE_loss = nn.BCELoss()

    def train(self, disc_iters=5, long_disc_iters=100, lambda_gp=10):
        """
        :param disc_iters:
        :param long_disc_iters:
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
                grad_outputs=torch.ones(output.shape, device=self.device),
                retain_graph=True,
                create_graph=True
            )[0]
            return ((grads.norm(2, dim=1) - 1) ** 2).mean()

        gen_iters = 0  # the generator is not trained every iteration
        for epoch in range(self.nr_epochs):
            for minibatch_iter, data in enumerate(self.dataloader):

                # the number of mini batches we'll train the critic before training the generator
                if gen_iters < 25 or gen_iters % 500 == 0:
                    D_iters = long_disc_iters
                else:
                    D_iters = disc_iters

                real = data.to(self.device)
                batch_size = real.size(0)

                """ Train the discriminator
                """

                self.optimizer_D.zero_grad()
                y = torch.ones(batch_size).to(self.device)
                pred = self.discriminator(real)
                lossD_real = self.BCE_loss(pred, y)
                lossD_real.backward()
                latent_zs = torch.randn(batch_size, self.nz, device=self.device)
                gen = self.generator(latent_zs).detach()
                y = y.fill_(0)
                pred_fake = self.discriminator(gen.detach())
                lossD_fake = self.BCE_loss(pred_fake, y)
                lossD_fake.backward()
                gradient_penalty = lambda_gp * _grad_penalty(real, gen)
                gradient_penalty.backward()
                loss_D = lossD_real + lossD_fake + gradient_penalty
                self.optimizer_D.step()

			    
                loss_G = None
                if self.global_iter % D_iters == 0:
                    """ Train the generator every [Diters]
                    """
                    self.optimizer_G.zero_grad()
                    y = y.fill_(1)
                    pred = self.discriminator(gen)
                    loss_G = self.BCE_loss(pred,y)
                    loss_G.backward()
                    self.optimizer_G.step()
                    gen_iters += 1

                # End iteration
                self._end_iteration(epoch, minibatch_iter, loss_G.item() if loss_G is not None else None, loss_D.item())

        return self.samples, self.D_losses, self.G_losses
