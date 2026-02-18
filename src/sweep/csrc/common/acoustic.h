#pragma once

struct AcousticWavefield{

    float* __restrict__ u_prev;
    float* __restrict__ u_now;
    float* __restrict__ u_next;

    float* __restrict__ psix;
    float* __restrict__ psiy;
    float* __restrict__ psiz;

    float* __restrict__ zetax;
    float* __restrict__ zetay;
    float* __restrict__ zetaz;

};

struct AcousticCPML{

    const float* __restrict__ ax;
    const float* __restrict__ bx;
    const float* __restrict__ dbxdx;

    const float* __restrict__ ay;
    const float* __restrict__ by;
    const float* __restrict__ dbydy;

    const float* __restrict__ az;
    const float* __restrict__ bz;
    const float* __restrict__ dbzdz;

};