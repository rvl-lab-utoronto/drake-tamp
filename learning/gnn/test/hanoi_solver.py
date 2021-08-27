
PEGS = ["blue_peg", "red_peg", "green_peg"]


def solve_hanoi_trad(start_peg, end_peg, discs):
    assert start_peg in PEGS
    assert end_peg in PEGS
    other_peg = (set(PEGS) - {start_peg, end_peg}).pop()
    num_discs = len(discs)
    if num_discs == 2:
        return [f"(move {discs[-1]} {start_peg} {other_peg})", f"(move {discs[-2]} {start_peg} {end_peg})", f"(move {discs[-1]} {other_peg} {end_peg})"]
    return solve_hanoi_trad(start_peg, other_peg, discs[1:]) + [f"(move {discs[0]} {start_peg} {end_peg})"] + solve_hanoi_trad(other_peg, end_peg, discs[1:])

def print_soln(soln):
    print(f"Length: {len(soln)}")
    for step in soln:
        print(step)

def solve_hanoi_real(start_peg, end_peg, discs):
    trad_soln = solve_hanoi_trad(start_peg, end_peg, discs)
    pegs = {p:[] for p in PEGS}
    disc_on = {d:d_on for d, d_on in zip(discs[1:], discs[:-1])}
    disc_on[discs[0]] = start_peg
    arm_at = "q_start"
    pegs[start_peg] = discs
    soln = []
    for motion in trad_soln:
        disc = motion.split(" ")[1]
        from_peg = motion.split(" ")[2]
        to_peg = motion.split(" ")[3][:-1]
        soln.append(f"(move-free {arm_at} {start_peg})")
        if disc_on[disc] in PEGS:
            soln.append(f"(pick {disc} {disc_on[disc]})")
        else:
            soln.append(f"(unstack {disc} {disc_on[disc]})")
        soln.append(f"(move-holding {disc} {from_peg} {to_peg})")
        if len(pegs[to_peg]) == 0:
            disc_on[disc] = to_peg
            soln.append(f"(place {disc} {to_peg})")
        else:
            disc_on[disc] = pegs[to_peg][-1]
            soln.append(f"(stack {disc} {pegs[to_peg][-1]})")
        pegs[to_peg].append(disc)
        pegs[from_peg].pop(-1)
        arm_at = to_peg

    return soln


if __name__ == "__main__":
    num_discs = 3
    discs = [f"disc_{num_discs - i}" for i in range(num_discs)]
    #soln =solve_hanoi_trad("red_peg", "blue_peg", discs)
    soln = solve_hanoi_real("red_peg", "blue_peg", discs)
    print_soln(soln)