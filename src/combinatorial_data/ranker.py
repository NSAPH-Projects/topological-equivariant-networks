def get_ranker(lifter_args: list[str]) -> callable:
    """
    Create a ranker function based on specified lifter arguments.

    The ranker function determines the rank of a cell based on its memberships,
    using either a specified nonnegative integer rank or cardinality ('c') as fallback.

    Parameters
    ----------
    lifter_args : list[str]
        A list of strings representing lifter arguments. Each string can be
        an identifier with an optional rank or 'c' for cardinality, separated
        by a colon (e.g., 'identity:c', 'ring:2').

    Returns
    -------
    callable
        A function that takes a cell and a list of membership booleans, returning
        an integer rank. The rank is the minimum among ranks derived from
        `lifter_args` for which the cell is a member.

    Raises
    ------
    ValueError
        If the requested rank is neither a nonnegative integer nor 'c'.

    Examples
    --------
    >>> ranker = get_ranker(['identity:c', 'ring:2'])
    >>> ranker(['a', 'b', 'c'], [True, False])
    2
    >>> ranker(['a', 'b'], [False, True])
    2
    >>> ranker(['a', 'b'], [True, True])
    1
    """
    ranking_logics = []
    for lifter in lifter_args:
        parts = lifter.split(":")

        # If no rank is specified, use cardinality as fallback
        try:
            lifter_rank = parts[1]
        except IndexError:
            ranking_logics.append("c")
            continue

        # Check if cardinality is requested
        if lifter_rank == "c":
            ranking_logics.append("c")
            continue

        # Try to convert the rank to an integer
        try:
            lifter_rank = int(lifter_rank)
        except ValueError:
            raise ValueError(f"Invalid rank '{lifter_rank}' specified for lifter '{parts[0]}'.")

        # Negative ranks are not allowed
        if lifter_rank < 0:
            raise ValueError(f"Negative cell ranks are not allowed, but you requested '{lifter}'.")

        # Add the rank to the list
        ranking_logics.append(lifter_rank)

    def ranker(cell: frozenset[int], memberships: list[bool]):
        """
        Determine the rank of a cell based on its memberships and predefined logics.

        If a cell is a member of multiple lifters, then its final rank will be the minimum among the
        ranks assigned to it by each lifter.

        Parameters
        ----------
        cell : frozenset[int]
            The cell to be ranked, assumed to be a collection or group of items.
        memberships : list[bool]
            A list indicating membership of the cell in various groups, corresponding to the
            `lifter_args` with which the ranker was created.

        Returns
        -------
        int
            The rank of the cell, determined as the minimum rank among the groups to which the cell
            belongs, with 'c' indicating rank based on cardinality.

        Raises
        ------
        ValueError
            If the length of `memberships` does not match the number of lifters.

        Notes
        -----
        The minimum is returned to avoid a situation where a singleton cell gets a nonzero rank,
        which is not allowed in TopoNetX.
        """
        if len(memberships) != len(lifter_args):
            raise ValueError(
                f"""Length of memberships ({len(memberships)}) does not match the number of lifters
                ({len(lifter_args)})."""
            )
        ranks = []
        for idx, is_member in enumerate(memberships):
            if is_member:
                if ranking_logics[idx] == "c":
                    ranks.append(len(cell) - 1)
                else:
                    ranks.append(ranking_logics[idx])
        return min(ranks)

    return ranker
