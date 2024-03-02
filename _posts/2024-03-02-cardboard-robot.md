---
layout: post
title: "Cardboard walking robot"
date: 2024-03-02T05:30:30-04:00
categories:
  - blog
tags:
  - diy
  - robotics
  - kids
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/TAdfJJIZrZw?si=9ME9YV8YY5OQhymJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

A while ago I was up in the middle of the night with a baby sleeping on me, deep in a YouTube rabbit hole about DIY electronics and robotics projects, and I came across a very cool design for a simple walking robot made out of cardboard and driven by a CD/DVD spindle motor.  Here's the original video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/Z7N0xCDVzIA?si=AIgw5YS9RADNyroP" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

The high-speed motor is geared down in two stages by cardboard-and-rubber-band pulleys to get some torque, and since the motion is rotary and constant the wiring is dead simple - you could probably even get away without doing any soldering, if you wanted

I thought it'd be a fun project and one that my 2-year old might get a kick out of, so I decided to give it a shot.
It was not as easy as I thought, so I figured I'd share a couple of lessons learned, in case you want to try to build one yourself.

#### 1. Buy the plans

The original design by "Blackfish" is available for something like $7 [on his website](https://bit.ly/3K87zSm) and I highly recommend getting them.
As I'll explain, there are a couple of somewhat tricky parts to the design.
I completely rebuilt it twice trying to reverse engineer it before I gave up and bought the plans, and then it worked fine on the third attempt.

![v1 (gearbox)](/assets/images/cardboard-robot/v1_gearbox.jpg){: width="250" }
![v1 (assembled)](/assets/images/cardboard-robot/v1_assembled.jpg){: width="250" }

One of the most important things is the different thicknesses of cardboard for different components, which I didn't even consider in my first attempts.

#### 2. Static stability

Most walking robots have to be carefully designed and controlled to remain stable.
Others, most famously the Boston Dynamics robots, lean into instabilities to perform acrobatic, natural motion.
The clever thing about this design, at least to me, is that it is _statically_ stable, meaning that no matter where in its "gait cycle" it is, you can turn the motor off and it will stand still.
This means that you don't really need any kind of controller or feedback stabilization; just a battery, a motor, and an on/off switch.

On the other hand, achieving that stability is a bit tricky - this is what I got wrong on my first attempt.
The critical dimension is the "shoulder width".
The wider the robot is, the more it's cantilevered on one leg, and the more torque is exerted about the ankle.
So better to build the torso as narrow as possible and still have room for all the components.

#### 3. Cardboard sucks

When I started building this I thought, "What a great idea! I'll prototype all kinds of stuff with cardboard."
By the end I thought, "I'm never building anything out of cardboard again."
By its nature, cardboard is pretty unforgiving.
The corrugation adds some stiffness in one dimension, but not the other.

In either direction, if you look at it wrong it will crease and then never be stiff again.
When I was done I gave it to my daughter, and after about 20 min she put it in the Baby Bjorn because "he needed to take a nap".
His foot got stuck, the leg bent, and he's never been able to walk unassisted since.

On the other hand, cardboard is something you probably have in your recycle bin, so there's that.
But if I was ever to try rebuilding again, I'd probably go with something like MDF.

![v2 (assembled)](/assets/images/cardboard-robot/v2_assembled.jpg){: width="250" }
![v2 (final)](/assets/images/cardboard-robot/v2_final.jpg){: width="250" }

The other trouble I had with materials was the popsicle sticks for the offsets on the axles.
I found it was kind of tricky to keep everything square when gluing those up, so I ended up cutting pieces off a paint-stirring stick like you get for free from Home Depot with a gallon of paint.

### Parts list

There are more details on the plans, but some of the links to AliExpress and such are broken, so here's what worked for me.
I got most of it from Adafruit. 
With this list it's all less than $50, and you can probably come up with cheaper or free substitutes for many things (like the pulleys).

- [Plans](https://blackfishspace.com/product/walking-robot-templates-pdf/) ($7)
- [CD/DVD Motor](https://www.adafruit.com/product/3882) ($2)
- [3.7V LiPo battery](https://www.adafruit.com/product/1570) ($6)
- [LiPo charger](https://www.adafruit.com/product/1304)  ($6)
- [JST connector](https://www.adafruit.com/product/261) ($1)
- [SPDT toggle switch](https://www.adafruit.com/product/3221) ($2)
- [Plastic pulleys and belts](https://www.amazon.com/dp/B083TGN78Y) ($12)
- [Cyanoacrylate super glue](https://www.amazon.com/dp/B004Y960MU) ($7)
- Bamboo kebab skewers
- Paint stick
- Hot glue gun
- Finger paint, stick-on jewels, and googly eyes (optional, but recommended)

Good luck!